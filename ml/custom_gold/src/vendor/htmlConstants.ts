export const REPLACEMENT_CHARACTER = "\uFFFD";

export const CODE_POINTS = {
    EOF: -1,
    NULL: 0,
    TABULATION: 9,
    CARRIAGE_RETURN: 13,
    LINE_FEED: 10,
    FORM_FEED: 12,
    SPACE: 32,
    EXCLAMATION_MARK: 33,
    QUOTATION_MARK: 34,
    NUMBER_SIGN: 35,
    AMPERSAND: 38,
    APOSTROPHE: 39,
    HYPHEN_MINUS: 45,
    SOLIDUS: 47,
    DIGIT_0: 48,
    DIGIT_9: 57,
    SEMICOLON: 59,
    LESS_THAN_SIGN: 60,
    EQUALS_SIGN: 61,
    GREATER_THAN_SIGN: 62,
    QUESTION_MARK: 63,
    LATIN_CAPITAL_A: 65,
    LATIN_CAPITAL_F: 70,
    LATIN_CAPITAL_X: 88,
    LATIN_CAPITAL_Z: 90,
    RIGHT_SQUARE_BRACKET: 93,
    GRAVE_ACCENT: 96,
    LATIN_SMALL_A: 97,
    LATIN_SMALL_F: 102,
    LATIN_SMALL_X: 120,
    LATIN_SMALL_Z: 122,
    REPLACEMENT_CHARACTER: 0xFFFD
};

export const CODE_POINT_SEQUENCES = {
    DASH_DASH_STRING: [45, 45],
    DOCTYPE_STRING: [68, 79, 67, 84, 89, 80, 69],
    CDATA_START_STRING: [91, 67, 68, 65, 84, 65, 91],
    SCRIPT_STRING: [115, 99, 114, 105, 112, 116],
    PUBLIC_STRING: [80, 85, 66, 76, 73, 67],
    SYSTEM_STRING: [83, 89, 83, 84, 69, 77]
};

export const ERRORS = {
    controlCharacterInInputStream: "control-character-in-input-stream",
    noncharacterInInputStream: "noncharacter-in-input-stream",
    surrogateInInputStream: "surrogate-in-input-stream",
    nonVoidHtmlElementStartTagWithTrailingSolidus: "non-void-html-element-start-tag-with-trailing-solidus",
    endTagWithAttributes: "end-tag-with-attributes",
    endTagWithTrailingSolidus: "end-tag-with-trailing-solidus",
    unexpectedSolidusInTag: "unexpected-solidus-in-tag",
    unexpectedNullCharacter: "unexpected-null-character",
    unexpectedQuestionMarkInsteadOfTagName: "unexpected-question-mark-instead-of-tag-name",
    invalidFirstCharacterOfTagName: "invalid-first-character-of-tag-name",
    unexpectedEqualsSignBeforeAttributeName: "unexpected-equals-sign-before-attribute-name",
    missingEndTagName: "missing-end-tag-name",
    unexpectedCharacterInAttributeName: "unexpected-character-in-attribute-name",
    unknownNamedCharacterReference: "unknown-named-character-reference",
    missingSemicolonAfterCharacterReference: "missing-semicolon-after-character-reference",
    unexpectedCharacterAfterDoctypeSystemIdentifier: "unexpected-character-after-doctype-system-identifier",
    unexpectedCharacterInUnquotedAttributeValue: "unexpected-character-in-unquoted-attribute-value",
    eofBeforeTagName: "eof-before-tag-name",
    eofInTag: "eof-in-tag",
    missingAttributeValue: "missing-attribute-value",
    missingWhitespaceBetweenAttributes: "missing-whitespace-between-attributes",
    missingWhitespaceAfterDoctypePublicKeyword: "missing-whitespace-after-doctype-public-keyword",
    missingWhitespaceBetweenDoctypePublicAndSystemIdentifiers: "missing-whitespace-between-doctype-public-and-system-identifiers",
    missingWhitespaceAfterDoctypeSystemKeyword: "missing-whitespace-after-doctype-system-keyword",
    missingQuoteBeforeDoctypePublicIdentifier: "missing-quote-before-doctype-public-identifier",
    missingQuoteBeforeDoctypeSystemIdentifier: "missing-quote-before-doctype-system-identifier",
    missingDoctypePublicIdentifier: "missing-doctype-public-identifier",
    missingDoctypeSystemIdentifier: "missing-doctype-system-identifier",
    abruptDoctypePublicIdentifier: "abrupt-doctype-public-identifier",
    abruptDoctypeSystemIdentifier: "abrupt-doctype-system-identifier",
    cdataInHtmlContent: "cdata-in-html-content",
    incorrectlyOpenedComment: "incorrectly-opened-comment",
    eofInScriptHtmlCommentLikeText: "eof-in-script-html-comment-like-text",
    eofInDoctype: "eof-in-doctype",
    nestedComment: "nested-comment",
    abruptClosingOfEmptyComment: "abrupt-closing-of-empty-comment",
    eofInComment: "eof-in-comment",
    incorrectlyClosedComment: "incorrectly-closed-comment",
    eofInCdata: "eof-in-cdata",
    absenceOfDigitsInNumericCharacterReference: "absence-of-digits-in-numeric-character-reference",
    nullCharacterReference: "null-character-reference",
    surrogateCharacterReference: "surrogate-character-reference",
    characterReferenceOutsideUnicodeRange: "character-reference-outside-unicode-range",
    controlCharacterReference: "control-character-reference",
    noncharacterCharacterReference: "noncharacter-character-reference",
    missingWhitespaceBeforeDoctypeName: "missing-whitespace-before-doctype-name",
    missingDoctypeName: "missing-doctype-name",
    invalidCharacterSequenceAfterDoctypeName: "invalid-character-sequence-after-doctype-name",
    duplicateAttribute: "duplicate-attribute",
    nonConformingDoctype: "non-conforming-doctype",
    missingDoctype: "missing-doctype",
    misplacedDoctype: "misplaced-doctype",
    endTagWithoutMatchingOpenElement: "end-tag-without-matching-open-element",
    closingOfElementWithOpenChildElements: "closing-of-element-with-open-child-elements",
    disallowedContentInNoscriptInHead: "disallowed-content-in-noscript-in-head",
    openElementsLeftAfterEof: "open-elements-left-after-eof",
    abandonedHeadElementChild: "abandoned-head-element-child",
    misplacedStartTagForHeadElement: "misplaced-start-tag-for-head-element",
    nestedNoscriptInHead: "nested-noscript-in-head",
    eofInElementThatCanContainOnlyText: "eof-in-element-that-can-contain-only-text"
};

export const NAMESPACES = {
    HTML: "http://www.w3.org/1999/xhtml",
    MATHML: "http://www.w3.org/1998/Math/MathML",
    SVG: "http://www.w3.org/2000/svg",
    XLINK: "http://www.w3.org/1999/xlink",
    XML: "http://www.w3.org/XML/1998/namespace",
    XMLNS: "http://www.w3.org/2000/xmlns/"
};

export const TAG_NAMES = {
    A: "a", ADDRESS: "address", APPLET: "applet", AREA: "area", ARTICLE: "article", ASIDE: "aside",
    BASE: "base", BASEFONT: "basefont", BGSOUND: "bgsound", BIG: "big", BLOCKQUOTE: "blockquote",
    BODY: "body", BR: "br", BUTTON: "button", CAPTION: "caption", CENTER: "center", COL: "col",
    COLGROUP: "colgroup", DD: "dd", DETAILS: "details", DIR: "dir", DIV: "div", DL: "dl", DT: "dt",
    EM: "em", EMBED: "embed", FIELDSET: "fieldset", FIGCAPTION: "figcaption", FIGURE: "figure",
    FONT: "font", FOOTER: "footer", FORM: "form", FRAME: "frame", FRAMESET: "frameset",
    H1: "h1", H2: "h2", H3: "h3", H4: "h4", H5: "h5", H6: "h6", HEAD: "head", HEADER: "header",
    HGROUP: "hgroup", HR: "hr", HTML: "html", I: "i", IFRAME: "iframe", IMG: "img", IMAGE: "image", INPUT: "input",
    KEYGEN: "keygen", LABEL: "label", LI: "li", LINK: "link", LISTING: "listing", MAIN: "main",
    MALIGNMARK: "malignmark", MARQUEE: "marquee", MATH: "math", MENU: "menu", META: "meta",
    MGLYPH: "mglyph", MI: "mi", MO: "mo", MN: "mn", MS: "ms", MTEXT: "mtext", NAV: "nav",
    NOBR: "nobr", NOEMBED: "noembed", NOFRAMES: "noframes", NOSCRIPT: "noscript",
    OBJECT: "object", OL: "ol", OPTGROUP: "optgroup", OPTION: "option", P: "p", PARAM: "param",
    PLAINTEXT: "plaintext", PRE: "pre", RB: "rb", RP: "rp", RT: "rt", RTC: "rtc", RUBY: "ruby",
    S: "s", SCRIPT: "script", SECTION: "section", SELECT: "select", SMALL: "small", SOURCE: "source",
    SPAN: "span", STRIKE: "strike", STRONG: "strong", STYLE: "style", SUB: "sub", SUMMARY: "summary",
    SUP: "sup", TABLE: "table", TBODY: "tbody", TD: "td", TEMPLATE: "template", TEXTAREA: "textarea",
    TFOOT: "tfoot", TH: "th", THEAD: "thead", TITLE: "title", TR: "tr", TRACK: "track", TT: "tt",
    U: "u", UL: "ul", VAR: "var", WBR: "wbr", XMP: "xmp",
    // MathML
    ANNOTATION_XML: "annotation-xml",
    // SVG
    SVG: "svg", FOREIGN_OBJECT: "foreignObject", DESC: "desc", TITLE_SVG: "title",
    DIALOG: "dialog"
};

export const ATTRS = {
    TYPE: "type",
    ACTION: "action",
    ENCTYPE: "enctype",
    METHOD: "method",
    NAME: "name",
    VALUE: "value",
    HIDDEN: "hidden",
    DISABLED: "disabled",
    READONLY: "readonly",
    SELECTED: "selected",
    CHECKED: "checked",
    MULTIPLE: "multiple",
    REQUIRED: "required",
    AUTOFOCUS: "autofocus",
    USEMAP: "usemap",
    ISMAP: "ismap",
    ALIGN: "align",
    VALIGN: "valign",
    BORDER: "border",
    WIDTH: "width",
    HEIGHT: "height",
    HSPACE: "hspace",
    VSPACE: "vspace",
    CHARSET: "charset",
    CONTENT: "content",
    HTTPEQUIV: "http-equiv",
    REL: "rel",
    HREF: "href",
    TARGET: "target",
    MEDIA: "media",
    COLOR: "color",
    FACE: "face",
    SIZE: "size",
    ENCODING: "encoding",
    PROMPT: "prompt"
};

export const DOCUMENT_MODE = {
    NO_QUIRKS: "no-quirks",
    QUIRKS: "quirks",
    LIMITED_QUIRKS: "limited-quirks"
};

export function isSurrogate(cp: number) {
    return cp >= 0xD800 && cp <= 0xDFFF;
}

export function isSurrogatePair(cp: number) {
    return cp >= 0xDC00 && cp <= 0xDFFF;
}

export function getSurrogatePairCodePoint(cp1: number, cp2: number) {
    return (cp1 - 0xD800) * 1024 + 0x2400 + cp2;
}

export function isControlCodePoint(cp: number) {
    return (cp !== 0x20 && cp !== 0x0A && cp !== 0x0D && cp !== 0x09 && cp !== 0x0C && cp >= 0x01 && cp <= 0x1F) || (cp >= 0x7F && cp <= 0x9F);
}

export function isUndefinedCodePoint(cp: number) {
    return (cp >= 0xFDD0 && cp <= 0xFDEF) || [
        0xFFFE, 0xFFFF, 0x1FFFE, 0x1FFFF, 0x2FFFE, 0x2FFFF, 0x3FFFE, 0x3FFFF,
        0x4FFFE, 0x4FFFF, 0x5FFFE, 0x5FFFF, 0x6FFFE, 0x6FFFF, 0x7FFFE, 0x7FFFF,
        0x8FFFE, 0x8FFFF, 0x9FFFE, 0x9FFFF, 0xAFFFE, 0xAFFFF, 0xBFFFE, 0xBFFFF,
        0xCFFFE, 0xCFFFF, 0xDFFFE, 0xDFFFF, 0xEFFFE, 0xEFFFF, 0xFFFFE, 0xFFFFF,
        0x10FFFE, 0x10FFFF
    ].indexOf(cp) > -1;
}

export const CHARACTER_REFERENCE_CODE_POINT_MAPPING = {
    128: 8364, 130: 8218, 131: 402, 132: 8222, 133: 8230, 134: 8224, 135: 8225,
    136: 710, 137: 8240, 138: 352, 139: 8249, 140: 338, 142: 381, 145: 8216,
    146: 8217, 147: 8220, 148: 8221, 149: 8226, 150: 8211, 151: 8212, 152: 732,
    153: 8482, 154: 353, 155: 8250, 156: 339, 158: 382, 159: 376
};

export const SVG_TAG_NAMES_ADJUSTMENT_MAP = {
    altglyph: "altGlyph", altglyphdef: "altGlyphDef", altglyphitem: "altGlyphItem",
    animatecolor: "animateColor", animatemotion: "animateMotion", animatetransform: "animateTransform",
    clippath: "clipPath", feblend: "feBlend", fecolormatrix: "feColorMatrix",
    fecomponenttransfer: "feComponentTransfer", fecomposite: "feComposite",
    feconvolvematrix: "feConvolveMatrix", fediffuselighting: "feDiffuseLighting",
    fedisplacementmap: "feDisplacementMap", fedistantlight: "feDistantLight",
    feflood: "feFlood", fefunca: "feFuncA", fefuncb: "feFuncB", fefuncg: "feFuncG",
    fefuncr: "feFuncR", fegaussianblur: "feGaussianBlur", feimage: "feImage",
    femerge: "feMerge", femergenode: "feMergeNode", femorphology: "feMorphology",
    feoffset: "feOffset", fepointlight: "fePointLight", fespecularlighting: "feSpecularLighting",
    fespotlight: "feSpotLight", fetile: "feTile", feturbulence: "feTurbulence",
    foreignobject: "foreignObject", glyphref: "glyphRef", lineargradient: "linearGradient",
    radialgradient: "radialGradient", textpath: "textPath"
};

export const SVG_ATTRS_ADJUSTMENT_MAP = {
    attributename: "attributeName", attributetype: "attributeType", basefrequency: "baseFrequency",
    baseprofile: "baseProfile", calcmode: "calcMode", clippathunits: "clipPathUnits",
    diffuseconstant: "diffuseConstant", edgemode: "edgeMode", filterunits: "filterUnits",
    glyphref: "glyphRef", gradienttransform: "gradientTransform", gradientunits: "gradientUnits",
    kernelmatrix: "kernelMatrix", kernelunitlength: "kernelUnitLength", keypoints: "keyPoints",
    keysplines: "keySplines", keytimes: "keyTimes", lengthadjust: "lengthAdjust",
    limitingconeangle: "limitingConeAngle", markerheight: "markerHeight", markerunits: "markerUnits",
    markerwidth: "markerWidth", maskcontentunits: "maskContentUnits", maskunits: "maskUnits",
    numoctaves: "numOctaves", pathlength: "pathLength", patterncontentunits: "patternContentUnits",
    patterntransform: "patternTransform", patternunits: "patternUnits", pointsatx: "pointsAtX",
    pointsaty: "pointsAtY", pointsatz: "pointsAtZ", preservealpha: "preserveAlpha",
    preserveaspectratio: "preserveAspectRatio", primitiveunits: "primitiveUnits", refx: "refX",
    refy: "refY", repeatcount: "repeatCount", repeatdur: "repeatDur",
    requiredextensions: "requiredExtensions", requiredfeatures: "requiredFeatures",
    specularconstant: "specularConstant", specularexponent: "specularExponent",
    spreadmethod: "spreadMethod", startoffset: "startOffset", stddeviation: "stdDeviation",
    stitchtiles: "stitchTiles", surfacescale: "surfaceScale", systemlanguage: "systemLanguage",
    tablevalues: "tableValues", targetx: "targetX", targety: "targetY", textlength: "textLength",
    viewbox: "viewBox", viewtarget: "viewTarget", xchannelselector: "xChannelSelector",
    ychannelselector: "yChannelSelector", zoomandpan: "zoomAndPan"
};

export const XML_ATTRS_ADJUSTMENT_MAP = {
    "xlink:actuate": { prefix: "xlink", name: "actuate", namespace: NAMESPACES.XLINK },
    "xlink:arcrole": { prefix: "xlink", name: "arcrole", namespace: NAMESPACES.XLINK },
    "xlink:href": { prefix: "xlink", name: "href", namespace: NAMESPACES.XLINK },
    "xlink:role": { prefix: "xlink", name: "role", namespace: NAMESPACES.XLINK },
    "xlink:show": { prefix: "xlink", name: "show", namespace: NAMESPACES.XLINK },
    "xlink:title": { prefix: "xlink", name: "title", namespace: NAMESPACES.XLINK },
    "xlink:type": { prefix: "xlink", name: "type", namespace: NAMESPACES.XLINK },
    "xml:base": { prefix: "xml", name: "base", namespace: NAMESPACES.XML },
    "xml:lang": { prefix: "xml", name: "lang", namespace: NAMESPACES.XML },
    "xml:space": { prefix: "xml", name: "space", namespace: NAMESPACES.XML },
    xmlns: { prefix: "", name: "xmlns", namespace: NAMESPACES.XMLNS },
    "xmlns:xlink": { prefix: "xmlns", name: "xlink", namespace: NAMESPACES.XMLNS }
};

export const QUIRKS_MODE_PUBLIC_IDS = [
    "+//silmaril//dtd html pro v0r11 19970101//",
    "-//as//dtd html 3.0 aswedit + extensions//",
    "-//advasoft ltd//dtd html 3.0 aswedit + extensions//",
    "-//ietf//dtd html 2.0 level 1//",
    "-//ietf//dtd html 2.0 level 2//",
    "-//ietf//dtd html 2.0 strict level 1//",
    "-//ietf//dtd html 2.0 strict level 2//",
    "-//ietf//dtd html 2.0 strict//",
    "-//ietf//dtd html 2.0//",
    "-//ietf//dtd html 2.1e//",
    "-//ietf//dtd html 3.0//",
    "-//ietf//dtd html 3.2 final//",
    "-//ietf//dtd html 3.2//",
    "-//ietf//dtd html 3//",
    "-//ietf//dtd html level 0//",
    "-//ietf//dtd html level 1//",
    "-//ietf//dtd html level 2//",
    "-//ietf//dtd html level 3//",
    "-//ietf//dtd html strict level 0//",
    "-//ietf//dtd html strict level 1//",
    "-//ietf//dtd html strict level 2//",
    "-//ietf//dtd html strict level 3//",
    "-//ietf//dtd html strict//",
    "-//ietf//dtd html//",
    "-//metrius//dtd metrius presentational//",
    "-//microsoft//dtd internet explorer 2.0 html strict//",
    "-//microsoft//dtd internet explorer 2.0 html//",
    "-//microsoft//dtd internet explorer 2.0 tables//",
    "-//microsoft//dtd internet explorer 3.0 html strict//",
    "-//microsoft//dtd internet explorer 3.0 html//",
    "-//microsoft//dtd internet explorer 3.0 tables//",
    "-//netscape comm. corp.//dtd html//",
    "-//netscape comm. corp.//dtd strict html//",
    "-//o'reilly and associates//dtd html 2.0//",
    "-//o'reilly and associates//dtd html extended 1.0//",
    "-//o'reilly and associates//dtd html extended relaxed 1.0//",
    "-//sq//dtd html 2.0 hotmetal + extensions//",
    "-//softquad software//dtd hotmetal pro 6.0::19990601::extensions to html 4.0//",
    "-//softquad//dtd hotmetal pro 4.0::19971010::extensions to html 4.0//",
    "-//spyglass//dtd html 2.0 extended//",
    "-//sun microsystems corp.//dtd hotjava html//",
    "-//sun microsystems corp.//dtd hotjava strict html//",
    "-//w3c//dtd html 3 1995-03-24//",
    "-//w3c//dtd html 3.2 draft//",
    "-//w3c//dtd html 3.2 final//",
    "-//w3c//dtd html 3.2//",
    "-//w3c//dtd html 3.2s draft//",
    "-//w3c//dtd html 4.0 frameset//",
    "-//w3c//dtd html 4.0 transitional//",
    "-//w3c//dtd html experimental 19960712//",
    "-//w3c//dtd html experimental 970421//",
    "-//w3c//dtd w3 html//",
    "-//w3o//dtd w3 html 3.0//",
    "-//webtechs//dtd mozilla html 2.0//",
    "-//webtechs//dtd mozilla html//"
];

export const QUIRKS_MODE_PUBLIC_IDS_EXTENSION = [
    "-//w3c//dtd html 4.01 frameset//",
    "-//w3c//dtd html 4.01 transitional//"
];

export const QUIRKS_MODE_SHORT_PUBLIC_IDS = [
    "-//w3o//dtd w3 html strict 3.0//en//",
    "-/w3c/dtd html 4.0 transitional/en",
    "html"
];

export const LIMITED_QUIRKS_MODE_PUBLIC_IDS = [
    "-//w3c//dtd xhtml 1.0 frameset//",
    "-//w3c//dtd xhtml 1.0 transitional//"
];

export function isWhitespace(cp: number) {
    return cp === CODE_POINTS.SPACE || cp === CODE_POINTS.LINE_FEED || cp === CODE_POINTS.TABULATION || cp === CODE_POINTS.FORM_FEED;
}

export function isDigit(cp: number) {
    return cp >= CODE_POINTS.DIGIT_0 && cp <= CODE_POINTS.DIGIT_9;
}

export function isUpperAlpha(cp: number) {
    return cp >= CODE_POINTS.LATIN_CAPITAL_A && cp <= CODE_POINTS.LATIN_CAPITAL_Z;
}

export function isLowerAlpha(cp: number) {
    return cp >= CODE_POINTS.LATIN_SMALL_A && cp <= CODE_POINTS.LATIN_SMALL_Z;
}

export function isAlpha(cp: number) {
    return isLowerAlpha(cp) || isUpperAlpha(cp);
}

export function isAlphaNumeric(cp: number) {
    return isAlpha(cp) || isDigit(cp);
}

export function isUpperHexDigit(cp: number) {
    return cp >= CODE_POINTS.LATIN_CAPITAL_A && cp <= CODE_POINTS.LATIN_CAPITAL_F;
}

export function isLowerHexDigit(cp: number) {
    return cp >= CODE_POINTS.LATIN_SMALL_A && cp <= CODE_POINTS.LATIN_SMALL_F;
}

export function isHexDigit(cp: number) {
    return isDigit(cp) || isUpperHexDigit(cp) || isLowerHexDigit(cp);
}

export function toLowerCodePoint(cp: number) {
    return cp + 0x20;
}

export function getUnicodeChar(cp: number) {
    if (cp <= 0xFFFF) return String.fromCharCode(cp);
    cp -= 0x10000;
    return String.fromCharCode((cp >>> 10 & 0x3FF) | 0xD800) + String.fromCharCode(0xDC00 | (cp & 0x3FF));
}

export function getLowerChar(cp: number) {
    return String.fromCharCode(toLowerCodePoint(cp));
}

export const SPECIAL_ELEMENTS = {
    [NAMESPACES.HTML]: {
        [TAG_NAMES.ADDRESS]: true, [TAG_NAMES.APPLET]: true, [TAG_NAMES.AREA]: true,
        [TAG_NAMES.ARTICLE]: true, [TAG_NAMES.ASIDE]: true, [TAG_NAMES.BASE]: true,
        [TAG_NAMES.BASEFONT]: true, [TAG_NAMES.BGSOUND]: true, [TAG_NAMES.BLOCKQUOTE]: true,
        [TAG_NAMES.BODY]: true, [TAG_NAMES.BR]: true, [TAG_NAMES.BUTTON]: true,
        [TAG_NAMES.CAPTION]: true, [TAG_NAMES.CENTER]: true, [TAG_NAMES.COL]: true,
        [TAG_NAMES.COLGROUP]: true, [TAG_NAMES.DD]: true, [TAG_NAMES.DETAILS]: true,
        [TAG_NAMES.DIR]: true, [TAG_NAMES.DIV]: true, [TAG_NAMES.DL]: true,
        [TAG_NAMES.DT]: true, [TAG_NAMES.EMBED]: true, [TAG_NAMES.FIELDSET]: true,
        [TAG_NAMES.FIGCAPTION]: true, [TAG_NAMES.FIGURE]: true, [TAG_NAMES.FOOTER]: true,
        [TAG_NAMES.FORM]: true, [TAG_NAMES.FRAME]: true, [TAG_NAMES.FRAMESET]: true,
        [TAG_NAMES.H1]: true, [TAG_NAMES.H2]: true, [TAG_NAMES.H3]: true,
        [TAG_NAMES.H4]: true, [TAG_NAMES.H5]: true, [TAG_NAMES.H6]: true,
        [TAG_NAMES.HEAD]: true, [TAG_NAMES.HEADER]: true, [TAG_NAMES.HGROUP]: true,
        [TAG_NAMES.HR]: true, [TAG_NAMES.HTML]: true, [TAG_NAMES.IFRAME]: true,
        [TAG_NAMES.IMG]: true, [TAG_NAMES.INPUT]: true, [TAG_NAMES.LI]: true,
        [TAG_NAMES.LINK]: true, [TAG_NAMES.LISTING]: true, [TAG_NAMES.MAIN]: true,
        [TAG_NAMES.MARQUEE]: true, [TAG_NAMES.MENU]: true, [TAG_NAMES.META]: true,
        [TAG_NAMES.NAV]: true, [TAG_NAMES.NOEMBED]: true, [TAG_NAMES.NOFRAMES]: true,
        [TAG_NAMES.NOSCRIPT]: true, [TAG_NAMES.OBJECT]: true, [TAG_NAMES.OL]: true,
        [TAG_NAMES.P]: true, [TAG_NAMES.PARAM]: true, [TAG_NAMES.PLAINTEXT]: true,
        [TAG_NAMES.PRE]: true, [TAG_NAMES.SCRIPT]: true, [TAG_NAMES.SECTION]: true,
        [TAG_NAMES.SELECT]: true, [TAG_NAMES.SOURCE]: true, [TAG_NAMES.STYLE]: true,
        [TAG_NAMES.SUMMARY]: true, [TAG_NAMES.TABLE]: true, [TAG_NAMES.TBODY]: true,
        [TAG_NAMES.TD]: true, [TAG_NAMES.TEMPLATE]: true, [TAG_NAMES.TEXTAREA]: true,
        [TAG_NAMES.TFOOT]: true, [TAG_NAMES.TH]: true, [TAG_NAMES.THEAD]: true,
        [TAG_NAMES.TITLE]: true, [TAG_NAMES.TR]: true, [TAG_NAMES.TRACK]: true,
        [TAG_NAMES.UL]: true, [TAG_NAMES.WBR]: true, [TAG_NAMES.XMP]: true
    },
    [NAMESPACES.MATHML]: {
        [TAG_NAMES.MI]: true, [TAG_NAMES.MO]: true, [TAG_NAMES.MN]: true,
        [TAG_NAMES.MS]: true, [TAG_NAMES.MTEXT]: true, [TAG_NAMES.ANNOTATION_XML]: true
    },
    [NAMESPACES.SVG]: {
        [TAG_NAMES.TITLE_SVG]: true, [TAG_NAMES.FOREIGN_OBJECT]: true, [TAG_NAMES.DESC]: true
    }
};
