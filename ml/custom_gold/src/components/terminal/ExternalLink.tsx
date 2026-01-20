
// External link helper for terminal output

import React from "react";
import { Link } from "../../vendor/inkLink.js";

export default function ExternalLink({ url, label }: { url: string; label?: string }) {
    const text = label || url;

    return <Link url={url}>{text}</Link>;
}
