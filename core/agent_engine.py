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

def get_active_model_for_agent():
    models = get_loaded_models()
    if models:
        return models[0]
    return "default"

def generate_ai_summary(instruction: str, diff_text: str, modified_files: list[str], model_id: str) -> dict:
    """Uses the local model to generate a standardized commit message and bullet summary."""
    if not modified_files or not diff_text:
        return {
            "commit_msg": f"AI Update: {instruction[:60]}",
            "summary": "Conversational inquiry / advisory response."
        }
    
    summary_prompt = (
        f"You are an automated Git release summarizer.\n\n"
        f"User Instruction: {instruction}\n"
        f"Modified Files: {', '.join(modified_files)}\n"
        f"Git Diff Summary:\n```\n{diff_text[:1500]}\n```\n\n"
        "Generate two things:\n"
        "1. A concise conventional commit message (max 65 chars, e.g. 'feat(core): add multi-file diff parser').\n"
        "2. A brief 1-2 sentence summary of what was changed.\n\n"
        "Format output strictly as:\n"
        "COMMIT_MSG: <message>\n"
        "SUMMARY: <summary>"
    )

    try:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": summary_prompt}],
            "temperature": 0.1,
            "max_tokens": 150
        }
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=20)
        if resp.status_code == 200:
            out = resp.json()["choices"][0]["message"]["content"].strip()
            c_match = re.search(r'COMMIT_MSG:\s*(.+)', out)
            s_match = re.search(r'SUMMARY:\s*([\s\S]+)', out)
            
            commit_msg = c_match.group(1).strip() if c_match else f"Update {', '.join(modified_files[:2])}"
            summary = s_match.group(1).strip() if s_match else instruction[:100]
            return {"commit_msg": commit_msg, "summary": summary}
    except Exception:
        pass

    return {
        "commit_msg": f"Update {', '.join(modified_files[:2])}: {instruction[:45]}",
        "summary": instruction
    }

def process_agent_task(repo_dir_name: str, target_files: list[str], instruction: str, thread_id: str = "", model_id: str = ""):
    workspace_path = os.path.join(WORKSPACES_ROOT, repo_dir_name)
    if not os.path.exists(workspace_path):
        return {"status": "error", "message": "Workspace not found"}

    # 1. Package context files
    context_blocks = []
    for rel_file in target_files:
        full_p = os.path.join(workspace_path, rel_file)
        if os.path.exists(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    content = f.read()
                context_blocks.append(f"### File: {rel_file}\n```\n{content}\n```")
            except Exception: pass

    context_str = "\n\n".join(context_blocks) if context_blocks else "No existing files selected as context."

    system_prompt = (
        "You are an expert AI software architect.\n"
        "Output complete updated or newly created files.\n"
        "Format EACH file output strictly as:\n"
        "### File: <relative_path>\n"
        "```\n"
        "<full file content without truncation>\n"
        "```"
    )

    user_prompt = (
        f"Active Workspace Context Files:\n\n{context_str}\n\n"
        f"Task Instruction:\n{instruction}\n\n"
        "Provide the complete implementation:"
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
        
        resp = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=150)
        if resp.status_code != 200:
            return {"status": "error", "message": f"LM Studio API Error: {resp.text}"}

        ai_response = resp.json()["choices"][0]["message"]["content"]
        
        # 2. Extract and write files to disk
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

        # 3. Retrieve working tree diff & generate AI summary
        git_status = get_workspace_git_status(repo_dir_name)
        ai_summary_meta = generate_ai_summary(instruction, git_status["diff"], modified_files, selected_model)

        # 4. Save persistently to .lmstudio_history.json
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
            "proposed_commit_msg": ai_summary_meta["commit_msg"],
            "summary": ai_summary_meta["summary"]
        }
        target_thread["messages"].append(msg_payload)

        # Append to unified timeline events list
        hist.setdefault("timeline_events", []).insert(0, {
            "id": f"evt-{int(datetime.datetime.utcnow().timestamp()*1000)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "type": "code_edit" if modified_files else "inquiry",
            "instruction": instruction,
            "summary": ai_summary_meta["summary"],
            "commit_msg": ai_summary_meta["commit_msg"],
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