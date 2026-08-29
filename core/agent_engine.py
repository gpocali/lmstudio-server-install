"""
Madison Agent Engine
Handles autonomous multi-file generation, Git diff analysis, and commit message synthesis.
"""

import os
import re
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

def process_agent_task(
    repo_dir_name: str, 
    target_files: list = None, 
    instruction: str = "", 
    thread_id: str = "", 
    persona_id: str = "coding", 
    custom_system_prompt: str = "", 
    model_id: str = "",
    **kwargs
):
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return {"status": "error", "message": "Workspace not found"}

    target_files = target_files or []

    # Auto-Reference Engine: If no files selected, scan instruction for file mentions
    if not target_files and instruction:
        all_disk_files = []
        for root, _, files in os.walk(workspace_path):
            if ".git" in root or "__pycache__" in root: continue
            for f in files:
                if f == ".lmstudio_history.json": continue
                rel_p = os.path.relpath(os.path.join(root, f), workspace_path)
                all_disk_files.append(rel_p)

        for df in all_disk_files:
            # Check if filename or base name appears in instruction
            base_name = os.path.basename(df)
            if df.lower() in instruction.lower() or base_name.lower() in instruction.lower():
                if df not in target_files:
                    target_files.append(df)

    context_blocks = []
    for rel_file in target_files:
        full_p = os.path.join(workspace_path, rel_file)
        if os.path.exists(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    content = f.read()
                # Truncate extremely large files to prevent token overflow (~12,000 chars per file)
                if len(content) > 12000:
                    content = content[:12000] + "\n... [File truncated for context limit] ..."
                context_blocks.append(f"### File: {rel_file}\n```\n{content}\n```")
            except Exception: pass

    context_str = "\n\n".join(context_blocks) if context_blocks else "No existing files selected as context."

    system_prompt = get_persona_prompt(persona_id, custom_system_prompt, domain="coding")
    user_prompt = (
        f"Active Workspace Context Files:\n\n{context_str}\n\n"
        f"Task Instruction:\n{instruction}\n\n"
        "Provide the complete implementation following the file modification format:"
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
        
        # Extended timeout to 300 seconds for large file generation
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=300)
        if resp.status_code != 200:
            return {"status": "error", "message": f"LM Studio API Error ({resp.status_code}): {resp.text[:300]}"}

        ai_response = resp.json()["choices"][0]["message"]["content"]
        
        file_pattern = re.compile(r'###\s*File:\s*([^\n\r]+)[\r\n]+\x60\x60\x60(?:[a-zA-Z0-9_\-]+)?[\r\n]+([\s\S]*?)[\r\n]+\x60\x60\x60', re.MULTILINE)
        matches = file_pattern.findall(ai_response)
        
        modified_files = []
        if matches:
            for file_rel_path, file_content in matches:
                clean_rel = file_rel_path.strip().lstrip("/")
                dest_file_path = os.path.join(workspace_path, clean_rel)
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                with open(dest_file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
                modified_files.append(clean_rel)
        elif len(target_files) == 1:
            fallback_pattern = re.compile(r'\x60\x60\x60(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)\n\x60\x60\x60')
            code_match = fallback_pattern.search(ai_response)
            if code_match:
                single_rel = target_files[0]
                dest_p = os.path.join(workspace_path, single_rel)
                with open(dest_p, "w", encoding="utf-8") as f:
                    f.write(code_match.group(1))
                modified_files.append(single_rel)

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
            "ai_response": ai_response,
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
    except Exception as e:
        return {"status": "error", "message": f"Execution error: {str(e)}"}