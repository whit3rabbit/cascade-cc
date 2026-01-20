export interface IdeDefinition {
    ideKind: 'vscode' | 'jetbrains';
    displayName: string;
    processKeywordsMac: string[];
    processKeywordsWindows: string[];
    processKeywordsLinux: string[];
}

export const IDE_REGISTRY: Record<string, IdeDefinition> = {
    cursor: {
        ideKind: "vscode",
        displayName: "Cursor",
        processKeywordsMac: ["Cursor Helper", "Cursor.app"],
        processKeywordsWindows: ["cursor.exe"],
        processKeywordsLinux: ["cursor"]
    },
    windsurf: {
        ideKind: "vscode",
        displayName: "Windsurf",
        processKeywordsMac: ["Windsurf Helper", "Windsurf.app"],
        processKeywordsWindows: ["windsurf.exe"],
        processKeywordsLinux: ["windsurf"]
    },
    vscode: {
        ideKind: "vscode",
        displayName: "VS Code",
        processKeywordsMac: ["Visual Studio Code", "Code Helper"],
        processKeywordsWindows: ["code.exe"],
        processKeywordsLinux: ["code"]
    },
    intellij: {
        ideKind: "jetbrains",
        displayName: "IntelliJ IDEA",
        processKeywordsMac: ["IntelliJ IDEA"],
        processKeywordsWindows: ["idea64.exe"],
        processKeywordsLinux: ["idea", "intellij"]
    },
    pycharm: {
        ideKind: "jetbrains",
        displayName: "PyCharm",
        processKeywordsMac: ["PyCharm"],
        processKeywordsWindows: ["pycharm64.exe"],
        processKeywordsLinux: ["pycharm"]
    },
    webstorm: {
        ideKind: "jetbrains",
        displayName: "WebStorm",
        processKeywordsMac: ["WebStorm"],
        processKeywordsWindows: ["webstorm64.exe"],
        processKeywordsLinux: ["webstorm"]
    },
    phpstorm: {
        ideKind: "jetbrains",
        displayName: "PhpStorm",
        processKeywordsMac: ["PhpStorm"],
        processKeywordsWindows: ["phpstorm64.exe"],
        processKeywordsLinux: ["phpstorm"]
    },
    rubymine: {
        ideKind: "jetbrains",
        displayName: "RubyMine",
        processKeywordsMac: ["RubyMine"],
        processKeywordsWindows: ["rubymine64.exe"],
        processKeywordsLinux: ["rubymine"]
    },
    clion: {
        ideKind: "jetbrains",
        displayName: "CLion",
        processKeywordsMac: ["CLion"],
        processKeywordsWindows: ["clion64.exe"],
        processKeywordsLinux: ["clion"]
    },
    goland: {
        ideKind: "jetbrains",
        displayName: "GoLand",
        processKeywordsMac: ["GoLand"],
        processKeywordsWindows: ["goland64.exe"],
        processKeywordsLinux: ["goland"]
    },
    rider: {
        ideKind: "jetbrains",
        displayName: "Rider",
        processKeywordsMac: ["Rider"],
        processKeywordsWindows: ["rider64.exe"],
        processKeywordsLinux: ["rider"]
    },
    datagrip: {
        ideKind: "jetbrains",
        displayName: "DataGrip",
        processKeywordsMac: ["DataGrip"],
        processKeywordsWindows: ["datagrip64.exe"],
        processKeywordsLinux: ["datagrip"]
    },
    appcode: {
        ideKind: "jetbrains",
        displayName: "AppCode",
        processKeywordsMac: ["AppCode"],
        processKeywordsWindows: ["appcode.exe"],
        processKeywordsLinux: ["appcode"]
    },
    dataspell: {
        ideKind: "jetbrains",
        displayName: "DataSpell",
        processKeywordsMac: ["DataSpell"],
        processKeywordsWindows: ["dataspell64.exe"],
        processKeywordsLinux: ["dataspell"]
    },
    aqua: {
        ideKind: "jetbrains",
        displayName: "Aqua",
        processKeywordsMac: [],
        processKeywordsWindows: ["aqua64.exe"],
        processKeywordsLinux: []
    },
    gateway: {
        ideKind: "jetbrains",
        displayName: "Gateway",
        processKeywordsMac: [],
        processKeywordsWindows: ["gateway64.exe"],
        processKeywordsLinux: []
    },
    fleet: {
        ideKind: "jetbrains",
        displayName: "Fleet",
        processKeywordsMac: [],
        processKeywordsWindows: ["fleet.exe"],
        processKeywordsLinux: []
    },
    androidstudio: {
        ideKind: "jetbrains",
        displayName: "Android Studio",
        processKeywordsMac: ["Android Studio"],
        processKeywordsWindows: ["studio64.exe"],
        processKeywordsLinux: ["android-studio"]
    }
};

export const JETBRAINS_FOLDER_MAP: Record<string, string[]> = {
    pycharm: ["PyCharm"],
    intellij: ["IntelliJIdea", "IdeaIC"],
    webstorm: ["WebStorm"],
    phpstorm: ["PhpStorm"],
    rubymine: ["RubyMine"],
    clion: ["CLion"],
    goland: ["GoLand"],
    rider: ["Rider"],
    datagrip: ["DataGrip"],
    appcode: ["AppCode"],
    dataspell: ["DataSpell"],
    aqua: ["Aqua"],
    gateway: ["Gateway"],
    fleet: ["Fleet"],
    androidstudio: ["AndroidStudio"]
};

export const CLI_DISPLAY_NAMES: Record<string, string> = {
    code: "VS Code",
    cursor: "Cursor",
    windsurf: "Windsurf",
    antigravity: "Antigravity",
    vi: "Vim",
    vim: "Vim",
    nano: "nano",
    notepad: "Notepad",
    "start /wait notepad": "Notepad",
    emacs: "Emacs",
    subl: "Sublime Text",
    atom: "Atom"
};

