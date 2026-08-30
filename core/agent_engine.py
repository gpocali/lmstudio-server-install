"""
Madison Agent Engine
Handles autonomous multi-file generation, Git diff analysis, and commit message synthesis.
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
    tree_str = "\n".join([f"- {f}" for f in file_tree]) if file_tree else "No existing files in repository."

    system_prompt = (
        "You are an expert Principal Software Architect.\n"
        "Analyze the user's task instruction and the repository file tree.\n"
        "1. Formulate a concise, descriptive name for the change (e.g. Conventional Commit format like 'feat(auth): implement token refresh' or a clear change title).\n"
        "2. Identify exactly which files in the repository need to be modified or newly created.\n"
        "3. Provide a clear, step-by-step implementation plan summary.\n\n"
        "Output your response STRICTLY as a valid JSON object matching this schema:\n"
        "{\n"
        "  \"change_name\": \"feat(scope): concise title of change\",\n"
        "  \"target_files\": [\"path/to/file1.py\", \"path/to/file2.py\"],\n"
        "  \"plan_summary\": \"1. Step one details...\\n2. Step two details...\"\n"
        "}"
    )

    user_prompt = f"Repository File Tree:\n{tree_str}\n\nTask Instruction:\n{instruction}"
    selected_model = model_id or get_active_model_for_agent()

    fallback_change_name = instruction.strip().split("\n")[0][:60]
    if len(fallback_change_name) >= 60:
        fallback_change_name = fallback_change_name[:57] + "..."

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
        
        # Clean markdown code blocks if present
        clean_json_str = content
        if "```json" in clean_json_str:
            clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json_str:
            clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

        # Find outer JSON boundaries if surrounded by conversational text
        json_match = re.search(r'\{[\s\S]*\}', clean_json_str)
        if json_match:
            clean_json_str = json_match.group(0)

        plan_data = json.loads(clean_json_str)
        target_files = plan_data.get("target_files", [])
        if isinstance(target_files, str):
            target_files = [target_files]
        elif not isinstance(target_files, list):
            target_files = []

        # Sanitize target file paths
        sanitized_files = []
        for tf in target_files:
            if isinstance(tf, str):
                cleaned_tf = tf.strip().lstrip("./\\").replace("\\", "/")
                if cleaned_tf and cleaned_tf not in sanitized_files:
                    sanitized_files.append(cleaned_tf)

        change_name = plan_data.get("change_name", "").strip() or fallback_change_name
        plan_summary = plan_data.get("plan_summary", "").strip() or "Execute proposed code modifications."

        return {
            "status": "success",
            "change_name": change_name,
            "target_files": sanitized_files,
            "plan_summary": plan_summary
        }
    except Exception as e:
        # Smart fallback detection from file_tree
        auto_targets = []
        for f in file_tree:
            base_low = os.path.basename(f).lower()
            inst_low = instruction.lower()
            if base_low in inst_low or f.lower() in inst_low:
                auto_targets.append(f)
        if not auto_targets and file_tree:
            auto_targets = file_tree[:3]

        return {
            "status": "success",
            "change_name": fallback_change_name,
            "target_files": auto_targets,
            "plan_summary": f"Implementation Plan:\n1. Analyze task requirements for '{instruction.strip()}'.\n2. Update target files to apply changes.\n3. Validate syntax and review git diff."
        }

def execute_approved_plan(
    repo_dir_name: str,
    target_files: list,
    instruction: str,
    change_name: str = "",
    thread_id: str = "",
    persona_id: str = "coding",
    custom_system_prompt: str = "",
    model_id: str = "",
    **kwargs
):
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return {"status": "error", "message": "Workspace not found"}

    context_blocks = []
    for rel_file in target_files:
        full_p = os.path.join(workspace_path, rel_file)
        if os.path.exists(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > 16000:
                    content = content[:16000] + "\n... [File truncated for context limit] ..."
                context_blocks.append(f"### File: {rel_file}\n```\n{content}\n```")
            except Exception: pass

    context_str = "\n\n".join(context_blocks) if context_blocks else "No existing files selected as context."

    system_prompt = get_persona_prompt(persona_id, custom_system_prompt, domain="coding")
    user_prompt = (
        f"Target Files Existing Contents:\n\n{context_str}\n\n"
        f"Task Instruction & Approved Plan:\n{instruction}\n\n"
        "Provide the complete, updated file implementations using the format:\n"
        "### File: path/to/file\n```python\n[complete updated file content]\n```"
    )

    selected_model = model_id or get_active_model_for_agent()

    try:
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 16384
        }
        
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=300)
        if resp.status_code != 200:
            return {"status": "error", "message": f"LM Studio API Error: {resp.text[:300]}"}

        ai_response = resp.json()["choices"][0]["message"]["content"]
        
        file_pattern = re.compile(r'###\s*File:\s*([^\n\r]+)[\r\n]+\x60\x60\x60(?:[a-zA-Z0-9_\-]+)?[\r\n]+([\s\S]*?)[\r\n]+\x60\x60\x60', re.MULTILINE)
        matches = file_pattern.findall(ai_response)
        
        modified_files = []
        if matches:
            for file_rel_path, file_content in matches:
                clean_rel = file_rel_path.strip().lstrip("./\\").replace("\\", "/")
                dest_file_path = os.path.join(workspace_path, clean_rel)
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                with open(dest_file_path, "w", encoding="utf-8") as f:
                    f.write(file_content.strip() + "\n")
                modified_files.append(clean_rel)
        elif len(target_files) == 1:
            fallback_pattern = re.compile(r'\x60\x60\x60(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)\n\x60\x60\x60')
            code_match = fallback_pattern.search(ai_response)
            if code_match:
                single_rel = target_files[0].strip().lstrip("./\\").replace("\\", "/")
                dest_p = os.path.join(workspace_path, single_rel)
                os.makedirs(os.path.dirname(dest_p), exist_ok=True)
                with open(dest_p, "w", encoding="utf-8") as f:
                    f.write(code_match.group(1).strip() + "\n")
                modified_files.append(single_rel)

        git_status = get_workspace_git_status(repo_dir_name)
        ai_summary_meta = generate_commit_msg_from_diff(repo_dir_name, modified_files, selected_model)

        effective_commit_msg = change_name.strip() or ai_summary_meta.get("commit_msg", f"Update {', '.join(modified_files[:2])}")
        effective_summary = change_name.strip() or ai_summary_meta.get("summary", instruction)

        hist = load_workspace_history(repo_dir_name)
        t_id = thread_id or hist.get("active_thread_id", "thread-default")
        
        target_thread = None
        for t in hist.get("threads", []):
            if t["id"] == t_id:
                target_thread = t
                break
        
        thread_title = change_name.strip() or f"Thread: {instruction[:25]}..."
        if not target_thread:
            target_thread = {
                "id": t_id,
                "title": thread_title,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "messages": []
            }
            hist.setdefault("threads", []).append(target_thread)
        elif target_thread.get("title", "").startswith("Thread: ") or target_thread.get("title") == "General Task Thread":
            if change_name.strip():
                target_thread["title"] = change_name.strip()

        msg_payload = {
            "id": f"msg-{int(datetime.datetime.utcnow().timestamp()*1000)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "instruction": instruction,
            "change_name": change_name,
            "target_files": target_files,
            "modified_files": modified_files,
            "ai_response": ai_response,
            "diff": git_status["diff"],
            "proposed_commit_msg": effective_commit_msg,
            "summary": effective_summary
        }
        target_thread["messages"].append(msg_payload)

        hist.setdefault("timeline_events", []).insert(0, {
            "id": f"evt-{int(datetime.datetime.utcnow().timestamp()*1000)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "type": "code_edit" if modified_files else "inquiry",
            "instruction": instruction,
            "change_name": change_name,
            "summary": effective_summary,
            "commit_msg": effective_commit_msg,
            "modified_files": modified_files,
            "diff": git_status["diff"]
        })

        save_workspace_history(repo_dir_name, hist)

        return {
            "status": "success",
            "thread_id": t_id,
            "change_name": change_name,
            "message": msg_payload,
            "git_status": git_status
        }
    except Exception as e:
        return {"status": "error", "message": f"Execution error: {str(e)}"}

# Backwards compatibility alias for queue workers
def process_agent_task(
    repo_dir_name: str,
    target_files: list = None,
    instruction: str = "",
    change_name: str = "",
    thread_id: str = "",
    persona_id: str = "coding",
    custom_system_prompt: str = "",
    model_id: str = "",
    **kwargs
):
    return execute_approved_plan(
        repo_dir_name=repo_dir_name,
        target_files=target_files or [],
        instruction=instruction,
        change_name=change_name,
        thread_id=thread_id,
        persona_id=persona_id,
        custom_system_prompt=custom_system_prompt,
        model_id=model_id,
        **kwargs
    )