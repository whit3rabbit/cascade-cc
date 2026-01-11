/**
 * Basic AAdecode implementation adapted from JSimplifier
 */
class AAdecode {
    constructor() {
        this.b = [
            "(c^_^o)", "(ﾟΘﾟ)", "((o^_^o) - (ﾟΘﾟ))", "(o^_^o)", "(ﾟｰﾟ)",
            "((ﾟｰﾟ) + (ﾟΘﾟ))", "((o^_^o) +(o^_^o))", "((ﾟｰﾟ) + (o^_^o))",
            "((ﾟｰﾟ) + (ﾟｰﾟ))", "((ﾟｰﾟ) + (ﾟｰﾟ) + (ﾟΘﾟ))", "(ﾟДﾟ) .ﾟωﾟﾉ",
            "(ﾟДﾟ) .ﾟΘﾟﾉ", "(ﾟДﾟ) ['c']", "(ﾟДﾟ) .ﾟｰﾟﾉ", "(ﾟДﾟ) .ﾟДﾟﾉ", "(ﾟДﾟ) [ﾟΘﾟ]"
        ];
    }

    decode(text) {
        try {
            let t = text;
            for (let i = 0; i < this.b.length; i++) {
                const searchValue = this.b[i] + "+ ";
                const replacementValue = i <= 7 ? String(i) : i.toString(16);
                const regex = new RegExp(searchValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
                t = t.replace(regex, replacementValue);
            }
            t = t.replace("(ﾟДﾟ)[ﾟoﾟ]) (ﾟΘﾟ)) ('_');", "");
            // This is a simplified version; real AAdecode needs more robust parsing
            // But for signal detection, this often works.
            return t;
        } catch (e) {
            return text;
        }
    }
}

/**
 * Basic JJdecode detection and extraction logic
 */
function jjdecode(text) {
    const jjencodePatterns = [
        /\$=~\[\];/,
        /\$\$=\{___:\+\+\$,/,
        /\$\$\$=\(\$\[\$\]\+\"\"\)\[\$\]/,
        /[\$_]{3,}.*[\[\]]{2,}.*[\+\!]{2,}/
    ];

    const isJJ = jjencodePatterns.some(pattern => pattern.test(text));
    if (!isJJ) return text;

    // In a real scenario, we'd use the eval-based extraction from JSimplifier
    // For now, we flag it or use a simplified unmasker if possible.
    // Since we are integrating with webcrack, it often handles these better.
    return text;
}

module.exports = {
    AAdecode,
    jjdecode
};
