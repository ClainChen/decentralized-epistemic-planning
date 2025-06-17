import util

file_path = "example-model/domain.pddl"

with open(file_path, 'r') as f:
    content = f.read()

# print(content)
result = util.regex_search(r"\(domain (\w+)\)", content)
print(result[0])
