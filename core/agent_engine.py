"""
Madison Agent Engine (Chunked & Segmented Patching Workflow)
Handles workspace mapping, incremental code block replacement, and robust diff application.
"""

import os
import re
import json
import datetime
import requests
import subprocess
from core.github_vault import (
    WORKSPACES_ROOT,
    load_workspace_history,
    save_workspace_history,
    get_workspace_git_status
)
from core.hardware import get_loaded_models
from core.personas import get_persona_prompt

def get_active_model_for_agent():
    models = get_loaded_models()
    if models:
        return models[0]
    return "default"

def generate_commit_msg_from_diff(repo_dir_name: str, target_files: list = None, model_id: str = "") -> dict:
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return {"status": "error", "message": "Workspace not found"}

    cmd = ["git", "-C", workspace_path, "diff"]
    if target_files:
        cmd.extend(["--"] + target_files)
    
    diff_res = subprocess.run(cmd, capture_output=True, text=True)
    diff_text = diff_res.stdout.strip()

    if not diff_text:
        status_res = subprocess.run(["git", "-C", workspace_path, "status", "--porcelain"], capture_output=True, text=True)
        untracked = [l.strip() for l in status_res.stdout.splitlines() if l.strip() and not l.strip().endswith(".lmstudio_history.json")]
        if untracked:
            diff_text = "Newly created untracked files:\n" + "\n".join(untracked)
        else:
            return {"status": "success", "commit_msg": "chore: update workspace", "summary": "No changes detected"}

    selected_model = model_id or get_active_model_for_agent()

    prompt = (
        "You are an expert Git commit message generator.\n"
        "Analyze the following git diff and output a concise Conventional Commit message "
        "(max 65 chars, e.g. 'feat(auth): add JWT expiration refresh').\n\n"
        f"GIT DIFF:\n```\n{diff_text[:3000]}\n```\n\n"
        "Output strictly in this format:\n"
        "COMMIT_MSG: <concise message>\n"
        "SUMMARY: <1-2 sentence overview of changes>"
    )

    try:
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 160
        }
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=30)
        if resp.status_code == 200:
            out = resp.json()["choices"][0]["message"]["content"].strip()
            c_match = re.search(r'COMMIT_MSG:\s*(.+)', out)
            s_match = re.search(r'SUMMARY:\s*([\s\S]+)', out)
            
            commit_msg = c_match.group(1).strip() if c_match else "chore: update files"
            summary = s_match.group(1).strip() if s_match else "Updated workspace files"
            return {"status": "success", "commit_msg": commit_msg, "summary": summary}
    except Exception as e:
        return {"status": "error", "message": f"Summarizer error: {str(e)}"}

    return {"status": "success", "commit_msg": "chore: update workspace files", "summary": "Updated files on disk"}

def get_workspace_file_tree(workspace_path: str) -> list[str]:
    file_list = []
    for root, _, files in os.walk(workspace_path):
        if ".git" in root or "__pycache__" in root: continue
        for f in files:
            if f == ".lmstudio_history.json": continue
            rel = os.path.relpath(os.path.join(root, f), workspace_path)
            file_list.append(rel)
    file_list.sort()
    return file_list

def generate_agent_plan(repo_dir_name: str, instruction: str, model_id: str = "") -> dict:
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return {"status": "error", "message": "Workspace not found"}

    file_tree = get_workspace_file_tree(workspace_path)
    tree_str = "\n".join([f"- {f}" for f in file_tree])

    system_prompt = (
        "You are an expert Principal Software Architect.\n"
        "Analyze the user's task instruction and repository file tree. "
        "Formulate a precise implementation plan targeting specific files.\n"
        "Output strictly in JSON format:\n"
        "{\n  \"target_files\": [\"path/to/file.py\"],\n  \"plan_summary\": \"Step 1: ...\\nStep 2: ...\"\n}"
    )

    user_prompt = f"Repository File Tree:\n{tree_str}\n\nTask Instruction:\n{instruction}"
    selected_model = model_id or get_active_model_for_agent()

    try:
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=60)
        if resp.status_code != 200:
            return {"status": "error", "message": f"LLM Plan Error: {resp.text}"}

        content = resp.json()["choices"][0]["message"]["content"].strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        plan_data = json.loads(content)
        return {
            "status": "success",
            "target_files": plan_data.get("target_files", []),
            "plan_summary": plan_data.get("plan_summary", "Review plan before execution.")
        }
    except Exception as e:
        return {
            "status": "success",
            "target_files": file_tree[:3],
            "plan_summary": f"Executing requested task across core files."
        }

def execute_approved_plan(
    repo_dir_name: str,
    target_files: list,
    instruction: str,
    thread_id: str = "",
    persona_id: str = "coding",
    custom_system_prompt: str = "",
    model_id: str = "",
    **kwargs
):
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return {"status": "error", "message": "Workspace not found"}

    modified_files = []
    selected_model = model_id or get_active_model_for_agent()
    system_prompt = get_persona_prompt(persona_id, custom_system_prompt, domain="coding")

    # Chunked file processing: process one file at a time to maximize context window utilization and prevent token overflow cutoffs
    for rel_file in target_files:
        full_p = os.path.join(workspace_path, rel_file)
        file_content = ""
        if os.path.exists(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except Exception: pass

        user_prompt = (
            f"File to modify: {rel_file}\n"
            f"Existing Content:\n```\n{file_content}\n```\n\n"
            f"Task Instruction & Plan:\n{instruction}\n\n"
            "Provide the complete updated file content using this exact block format:\n"
            f"### File: {rel_file}\n```\n[complete updated file content]\n```"
        )

        try:
            payload = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 16384
            }
            
            resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=300)
            if resp.status_code != 200:
                continue

            ai_response = resp.json()["choices"][0]["message"]["content"]
            
            # Parse block
            file_pattern = re.compile(r'###\s*File:\s*' + re.escape(rel_file) + r'[\r\n]+\x60\x60\x60(?:[a-zA-Z0-9_\-]+)?[\r\n]+([\s\S]*?)[\r\n]+\x60\x60\x60', re.IGNORECASE)
            match = file_pattern.search(ai_response)
            
            if match:
                new_content = match.group(1).strip()
            else:
                # Fallback to general code block
                fb_pattern = re.compile(r'\x60\x60\x60(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)\n\x60\x60\x60')
                fb_match = fb_pattern.search(ai_response)
                new_content = fb_match.group(1).strip() if fb_match else ai_response.strip()

            if new_content and len(new_content) > 10:
                os.makedirs(os.path.dirname(full_p), exist_ok=True)
                with open(full_p, "w", encoding="utf-8") as f:
                    f.write(new_content + "\n")
                modified_files.append(rel_file)
        except Exception:
            pass

    git_status = get_workspace_git_status(repo_dir_name)
    ai_summary_meta = generate_commit_msg_from_diff(repo_dir_name, modified_files, selected_model)

    hist = load_workspace_history(repo_dir_name)
    t_id = thread_id or hist.get("active_thread_id", "thread-default")
    
    target_thread = None
    for t in hist.get("threads", []):
        if t["id"] == t_id:
            target_thread = t
            break
    
    if not target_thread:
        target_thread = {
            "id": t_id,
            "title": f"Thread: {instruction[:25]}...",
            "created_at": datetime.datetime.utcnow().isoformat(),
            "messages": []
        }
        hist.setdefault("threads", []).append(target_thread)

    msg_payload = {
        "id": f"msg-{int(datetime.datetime.utcnow().timestamp()*1000)}",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "instruction": instruction,
        "target_files": target_files,
        "modified_files": modified_files,
        "ai_response": f"Successfully chunked and updated {len(modified_files)} files.",
        "diff": git_status["diff"],
        "proposed_commit_msg": ai_summary_meta.get("commit_msg", f"Update {', '.join(modified_files[:2])}"),
        "summary": ai_summary_meta.get("summary", instruction)
    }
    target_thread["messages"].append(msg_payload)

    hist.setdefault("timeline_events", []).insert(0, {
        "id": f"evt-{int(datetime.datetime.utcnow().timestamp()*1000)}",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "type": "code_edit" if modified_files else "inquiry",
        "instruction": instruction,
        "summary": ai_summary_meta.get("summary", instruction),
        "commit_msg": ai_summary_meta.get("commit_msg", f"Update {', '.join(modified_files[:2])}"),
        "modified_files": modified_files,
        "diff": git_status["diff"]
    })

    save_workspace_history(repo_dir_name, hist)

    return {
        "status": "success",
        "thread_id": t_id,
        "message": msg_payload,
        "git_status": git_status
    }

def process_agent_task(repo_dir_name: str, target_files: list = None, instruction: str = "", thread_id: str = "", persona_id: str = "coding", custom_system_prompt: str = "", model_id: str = "", **kwargs):
    return execute_approved_plan(
        repo_dir_name=repo_dir_name,
        target_files=target_files or [],
        instruction=instruction,
        thread_id=thread_id,
        persona_id=persona_id,
        custom_system_prompt=custom_system_prompt,
        model_id=model_id,
        **kwargs
    )