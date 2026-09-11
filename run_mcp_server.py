# coding: utf-8
# A shortcut to launch OpenManus MCP server, where its introduction also solves other import issues.
from app.mcp.server import MCPServer, parse_args
from app.tool.policy import ToolPolicy


if __name__ == "__main__":
    args = parse_args()

    # Create and run server (maintaining original flow)
    policy = ToolPolicy.guarded(
        allowed_tools=args.allow_tool,
        allowed_tainted_tools=args.allow_tainted_tool,
        approved_operations=args.approve_tainted_operation,
        audit_log_path=args.audit_log,
    )
    server = MCPServer(tool_policy=policy)
    server.run(transport=args.transport)
