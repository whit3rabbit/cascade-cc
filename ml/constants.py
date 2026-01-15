# Common Babel node types.
NODE_TYPES = [
    "UNKNOWN", "File", "Program", "FunctionDeclaration", "FunctionExpression", 
    "ArrowFunctionExpression", "VariableDeclaration", "VariableDeclarator",
    "Identifier", "StringLiteral", "NumericLiteral", "BooleanLiteral",
    "NullLiteral", "RegExpLiteral", "BinaryExpression", "UnaryExpression",
    "UpdateExpression", "LogicalExpression", "AssignmentExpression",
    "MemberExpression", "OptionalMemberExpression", "CallExpression",
    "OptionalCallExpression", "NewExpression", "ArrayExpression",
    "ObjectExpression", "ObjectProperty", "ObjectMethod", "BlockStatement",
    "ExpressionStatement", "IfStatement", "ForStatement", "WhileStatement",
    "DoWhileStatement", "ForInStatement", "ForOfStatement", "ReturnStatement",
    "ThrowStatement", "TryStatement", "CatchClause", "SwitchStatement",
    "SwitchCase", "BreakStatement", "ContinueStatement", "EmptyStatement",
    "DebuggerStatement", "WithStatement", "LabeledStatement", "ClassDeclaration",
    "ClassBody", "ClassMethod", "ClassProperty", "ImportDeclaration",
    "ImportSpecifier", "ImportDefaultSpecifier", "ImportNamespaceSpecifier",
    "ExportNamedDeclaration", "ExportDefaultDeclaration", "ExportAllDeclaration",
    "YieldExpression", "AwaitExpression", "TemplateLiteral", "TemplateElement",
    "SpreadElement", "RestElement", "SequenceExpression", "AssignmentPattern",
    "ArrayPattern", "ObjectPattern", "V8IntrinsicIdentifier"
]

TYPE_TO_ID = {t: i for i, t in enumerate(NODE_TYPES)}
VOCAB_SIZE = len(NODE_TYPES)
