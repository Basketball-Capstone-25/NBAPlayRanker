/**
 * Infrastructure Layer barrel file.
 *
 * Subsystems:
 * - Auth: Supabase authentication gateway (signIn, signUp)
 * - API Client: HTTP helpers and type definitions for the FastAPI backend
 */
export { signIn, signUp, type UserRole, type SignInResult } from "./auth";
export * from "./api-client";
