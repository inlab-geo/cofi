# Helper script to link a new inference tool:
# $ python scripts/new_inference_tool.py <inference-tool-name>
# Replacing <inference-tool-name> with the new tool name.

import glob
import sys
import pathlib


current_dir = pathlib.Path(__file__).resolve().parent
root_dir = current_dir.parent
INFTOOL_FOLDER = str(root_dir / "src/cofi/tools")
TEMPLATE_FILE = str(root_dir / "scripts/_template/template_tool.py")


def main():
    print("Nice to see new inference tool getting in.")
    print("We are generating a new file from template...\n")

    # validate inference tool name
    if len(sys.argv) != 2:
        raise RuntimeError(
            "No inference tool name detected.\n\n"
            "Usage: python new_inference_tool.py <new_inference_tool_name>"
        )
    inf_tool_name = sys.argv[-1]
    existing_tools_paths = glob.glob(INFTOOL_FOLDER+"/*.py")
    existing_tools = set()
    for p in existing_tools_paths:
        if "__init__.py" in p or "_base_inference_tool.py" in p:
            continue
        existing_tools.add(p.split("/")[-1][1:-3])
    if inf_tool_name in existing_tools:
        raise ValueError(
            "The inference tool name provided already exists, "
            "please choose another name"
        )
    elif inf_tool_name in ["_init__", "base_inference_tool"]:
        raise ValueError(
            "This file name is occupied in `src/cofi/tools/, "
            "please choose another name"
        )
    
    # convert inference tool name to other formats
    inf_tool_class_name = inf_tool_name.title().replace("_", "").replace("-", "")
    inf_tool_file_name = f"_{inf_tool_name}.py"

    # generate new inference tool file
    with open(TEMPLATE_FILE, "r") as f:
        content = f.read()
    content = content.replace("TemplateTool", inf_tool_class_name)
    content = content.replace("template_tool", inf_tool_name)
    with open(f"{INFTOOL_FOLDER}/{inf_tool_file_name}", "w") as new_f:
        new_f.write(content)
    print("\n🎉 OK.")
    print(f"Please navigate to {INFTOOL_FOLDER}/{inf_tool_file_name} to continue developing.")
    print("Remember to resolve all the `FIXME` tasks before running tests.")


if __name__ == "__main__":
    main()
