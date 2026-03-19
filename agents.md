# Agents Workflow Rules

每次按用户要求完成并应用代码修改后，必须立即执行以下 Git 提交流程，不等待用户催促：

1. 先检查改动：`git diff`
2. 自动暂存全部改动：`git add .`
3. 根据本次具体修改内容编写详细 Commit Message，并执行提交：`git commit -m "<detailed message>"`

要求：

- Commit Message 必须具体描述改了哪些文件、做了什么变更、目的是什么。
- 每次代码修改都要完成一次提交。
