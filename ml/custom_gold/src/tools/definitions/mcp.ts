import { z } from "zod";

export const McpInputSchema = z.record(z.string(), z.unknown());
