import { useState, useEffect } from 'react';

export function useMcpClients(initialClients = []) {
    const [clients, setClients] = useState(initialClients);
    // Logic to manage clients
    return clients;
}
