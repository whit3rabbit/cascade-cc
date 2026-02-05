/**
 * File: src/services/mobile/QRCodeService.ts
 * Role: Generates terminal-friendly QR codes.
 */

import QRCode from 'qrcode';

/**
 * Generates a QR code string suitable for display in the terminal.
 * Uses UTF-8 block characters.
 */
export async function generateQRCode(text: string): Promise<string> {
    try {
        const result = await QRCode.toString(text, {
            type: 'utf8',
            errorCorrectionLevel: 'L',
            margin: 1,
            scale: 1,
            color: {
                dark: '#FFFFFF', // White blocks for data
                light: '#00000000' // Transparent background (or suitable for dark terminal)
            }
        });
        return result;
    } catch (error) {
        console.error('Failed to generate QR code:', error);
        return 'Error generating QR code.';
    }
}
