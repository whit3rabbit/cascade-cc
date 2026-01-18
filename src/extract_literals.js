const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;

function extractLiterals(code) {
    try {
        const ast = parser.parse(code, {
            sourceType: 'module',
            plugins: ['jsx', 'typescript']
        });
        const literals = new Set();
        traverse(ast, {
            StringLiteral(path) {
                if (path.node.value.length > 5) {
                    literals.add(path.node.value);
                }
            },
            TemplateLiteral(path) {
                path.node.quasis.forEach(q => {
                    if (q.value.raw.length > 5) {
                        literals.add(q.value.raw);
                    }
                });
            }
        });
        return Array.from(literals);
    } catch (e) {
        return [];
    }
}

module.exports = { extractLiterals };
