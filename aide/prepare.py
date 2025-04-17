from pathlib import Path
from typing import List
import json

def read_raw_markdown(file_path: Path) -> str:
    with open(file_path, "r", encoding='utf-8') as f:
        return "```\n" + f.read() + "\n```\n"

def read_jupyter_notebook(file_path: Path) -> str:
    """ Read the code and markdown cells from a Jupyter notebook. """

    assert file_path.suffix == ".ipynb", "File must be a Jupyter notebook."

    with open(file_path, "r", encoding='utf-8') as f:
        try:
            notebook = json.load(f)
        except json.JSONDecodeError:
            """ This indicates that the file is a raw markdown file. """
            return read_raw_markdown(file_path)

    result = ""
    last_cell_type = "markdown"
    for cell in notebook["cells"]:
        if last_cell_type != cell["cell_type"]:
            result += "\n```\n"
        if cell["cell_type"] == "code":
            result += "".join(cell["source"])
        elif cell["cell_type"] == "markdown":
            result += "".join(cell["source"])
        last_cell_type = cell["cell_type"]
    if last_cell_type == "code":
        result += "\n```\n"
    return result

def read_discussion(file_path: Path) -> str:
    assert file_path.suffix == ".json", "File must be a JSON file."

    def walk(obj) -> List[str]:
        comments_field = "comments" if "comments" in obj else "replies"
        if comments_field not in obj:
            return []
        result = []
        for comment in obj[comments_field]:
            if "isDeleted" in comment and comment["isDeleted"]:
                continue
            if 'author' not in comment or 'displayName' not in comment['author']:
                continue
            if 'tier' not in comment['author']:
                comment['author']['tier'] = 'N/A'
            result.append(f"    + ({comment['author']['displayName']} <TIER: {comment['author']['tier']}>) " + comment['rawMarkdown'])
            result += ["    " + line for line in walk(comment)]
        return result

    with open(file_path, "r") as f:
        discussion = json.load(f)['forumTopic']
    
    if 'authorPerformanceTier' not in discussion:
        discussion['authorPerformanceTier'] = 'N/A'
    result = [f"# {discussion['name']}"]
    result += [f"({discussion['authorUserDisplayName']} <TIER: {discussion['authorPerformanceTier']}>) " + discussion["firstMessage"]["rawMarkdown"]]
    result += walk(discussion)
    return "\n".join(result)

def read_config(file_path: Path, type: str="discussion") -> dict:
    """ Read the configuration file. """
    assert file_path.suffix == ".json", "File must be a JSON file."
    with open(file_path, "r") as f:
        config = json.load(f)
    
    result = {}
    for entry in config:
        if type == "discussion":
            result[str(entry["id"])] = {
                "title": entry["title"],
                "votes": entry["votes"] if "votes" in entry else 0,
                "id": str(entry["id"]),
            }
        elif type == "kernel":
            result[entry["scriptVersionId"]] = {
                "title": entry["title"],
                "votes": entry["totalVotes"] if "totalVotes" in entry else 0,
                "id": entry["scriptVersionId"],
            }
    return result
    
if __name__ == "__main__":
    path = Path("/home/planarg/aideml/aide/example_tasks/tabular-playground-series-may-2022/scripts")
    output_path = Path("/home/planarg/aideml/aide/example_tasks/tabular-playground-series-may-2022/docs/kernels")
    output_path.mkdir(parents=True, exist_ok=True)
    files = Path(path).glob("*")

    for file in files:
        content = None
        if file.suffix == ".json":
            content = read_discussion(file)
        elif file.suffix == ".ipynb":
            content = read_jupyter_notebook(file)
        
        if content:
            with open(output_path / f"{file.stem}.txt", "w", encoding='utf-8') as f:
                f.write(content)
                print(f"Saved {file} to {file.stem}.txt")
    
    with open(output_path / "info.json", "w") as f:
        json.dump(read_config(path.parent / "kernels.json", type="kernel"), f, indent=4)
        print(f"Saved {output_path / 'info.json'}")

        

        