export class IndentedWriter {
    content: string[];
    indent: number;
    args: any[];

    constructor(args: any[] = []) {
        this.content = [];
        this.indent = 0;
        this.args = args;
    }

    indented(fn: (writer: IndentedWriter) => void) {
        this.indent += 1;
        fn(this);
        this.indent -= 1;
    }

    write(content: string | ((writer: IndentedWriter, context: any) => void)) {
        if (typeof content === "function") {
            content(this, {
                execution: "sync"
            });
            content(this, {
                execution: "async"
            });
            return;
        }

        const lines = content.split("\n").filter((line) => line);
        if (lines.length === 0) return;

        const minIndent = Math.min(...lines.map((line) => line.length - line.trimStart().length));
        const indentedLines = lines.map((line) => line.slice(minIndent)).map((line) => " ".repeat(this.indent * 2) + line);

        for (const line of indentedLines) {
            this.content.push(line);
        }
    }

    compile(): Function {
        const fn = Function;
        const args = this.args;
        const body = [...(this.content ?? [""]).map((line) => `  ${line}`)];
        return new fn(...args, body.join("\n"));
    }
}
