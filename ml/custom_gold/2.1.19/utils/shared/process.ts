export function platform(): string {
    return process.platform;
}

export function terminal(): string {
    return process.env.TERM || 'unknown';
}
