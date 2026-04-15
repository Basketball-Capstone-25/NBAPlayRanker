/**
 * I-prefix interfaces for the Infrastructure Layer (Frontend).
 *
 * These interfaces define the public contracts for frontend infrastructure subsystems.
 */

export type UserRole = "coach" | "analyst";

export interface ISignInResult {
  userId: string;
  role: UserRole;
}

export interface IAuthGateway {
  signIn(email: string, password: string): Promise<ISignInResult>;
  signUp(email: string, password: string, role: UserRole): Promise<void>;
}
