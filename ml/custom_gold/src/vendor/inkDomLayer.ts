/**
 * Ink DOM layer implementation.
 * Deobfuscated from various symbols in chunk_192.ts.
 */

export interface InkNode {
    nodeName: string;
    style: any;
    attributes: Record<string, any>;
    childNodes: InkNode[];
    parentNode?: InkNode;
    yogaNode?: any; // Yoga.Node
    nodeValue?: string;
    dirty: boolean;
}

/**
 * Creates an Ink VDOM node.
 * Deobfuscated from UtA in chunk_192.ts.
 */
export function createInkNode(nodeName: string): InkNode {
    return {
        nodeName,
        style: {},
        attributes: {},
        childNodes: [],
        dirty: false
    };
}

/**
 * Appends a child to an Ink node.
 * Deobfuscated from wtA in chunk_192.ts.
 */
export function appendChild(parent: InkNode, child: InkNode): void {
    if (child.parentNode) removeChild(child.parentNode, child);
    child.parentNode = parent;
    parent.childNodes.push(child);
    markDirty(parent);
}

/**
 * Removes a child from an Ink node.
 * Deobfuscated from ONA in chunk_192.ts.
 */
export function removeChild(parent: InkNode, child: InkNode): void {
    child.parentNode = undefined;
    const index = parent.childNodes.indexOf(child);
    if (index >= 0) parent.childNodes.splice(index, 1);
    markDirty(parent);
}

/**
 * Sets an attribute on an Ink node.
 * Deobfuscated from jc1 in chunk_192.ts.
 */
export function setAttribute(node: InkNode, key: string, value: any): void {
    if (node.attributes[key] === value) return;
    node.attributes[key] = value;
    markDirty(node);
}

/**
 * Sets styles on an Ink node.
 * Deobfuscated from Tc1 in chunk_192.ts.
 */
export function setStyle(node: InkNode, style: any): void {
    node.style = style;
    markDirty(node);
}

/**
 * Creates an Ink text node.
 * Deobfuscated from QXB in chunk_192.ts.
 */
export function createTextNode(value: string): InkNode {
    return {
        nodeName: "#text",
        nodeValue: value,
        style: {},
        attributes: {},
        childNodes: [],
        dirty: false
    };
}

/**
 * Sets the value of an Ink text node.
 * Deobfuscated from MNA in chunk_192.ts.
 */
export function setTextValue(node: InkNode, value: string): void {
    if (node.nodeValue === value) return;
    node.nodeValue = value;
    markDirty(node);
}

/**
 * Extracts text and styles from an Ink node tree.
 * Deobfuscated from EtA in chunk_192.ts.
 */
export function extractTextStyles(node: InkNode, parentStyle: any = {}): { text: string; styles: any }[] {
    const styles = { ...parentStyle, ...node.style };
    const results: { text: string; styles: any }[] = [];

    for (const child of node.childNodes) {
        if (child.nodeName === "#text") {
            results.push({ text: child.nodeValue || "", styles });
        } else {
            results.push(...extractTextStyles(child, styles));
        }
    }

    return results;
}

function markDirty(node: InkNode): void {
    let current: InkNode | undefined = node;
    while (current) {
        current.dirty = true;
        current = current.parentNode;
    }
}
