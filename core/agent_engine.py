import os
import re
import requests
import subprocess
from core.github_vault import WORKSPACES_ROOT
from core.hardware import get_loaded_models

def get_active_model_for_agent():
    models = get_loaded_models()
    if models:
        return models[0]
    return "default"

def process_agent_task(repo_dir_name: str, target_files: list[str], instruction: str, model_id: str = ""):
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

        # Get working tree diff (unstaged/uncommitted changes)
        diff_res = subprocess.run(["git", "-C", workspace_path, "diff"], capture_output=True, text=True)
        diff_text = diff_res.stdout or ""
        
        # If diff is empty (e.g. newly created untracked files), check status
        if not diff_text and modified_files:
            diff_text = f"New file(s) staged on disk:\n" + "\n".join([f"+ {f}" for f in modified_files])

        proposed_commit_msg = f"Update {', '.join(modified_files[:3])}: {instruction[:50]}" if modified_files else "Code update"

        return {
            "status": "success",
            "modified_files": modified_files,
            "proposed_commit_msg": proposed_commit_msg,
            "diff": diff_text[:5000],
            "ai_response": ai_response
        }
    except Exception as e:
        return {"status": "error", "message": f"Execution error: {str(e)}"}