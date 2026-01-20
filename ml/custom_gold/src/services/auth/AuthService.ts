import { updateSettings } from "../terminal/settings.js";

export const AuthService = {
    async setupToken(token: string) {
        // In the original, this might involve verifying the token with an API
        // For now, we just save it to the user settings.
        updateSettings("userSettings", {
            oauthTokenFromFd: token
        });
        return { success: true };
    }
};
