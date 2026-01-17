const isProduction = process.env.NODE_ENV === "production";

// Fallback to the live backend URL if environment variable is missing in production
// Fallback to localhost:8001 for local development
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (isProduction
    ? "https://cathode-backend-o4js3vzl2a-uc.a.run.app"
    : "http://localhost:8000");

export const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
export default API_BASE_URL;
