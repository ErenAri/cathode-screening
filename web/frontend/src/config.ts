const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

console.log("----------------------------------------");
console.log("🔧 Application Configuration Debug:");
console.log("   API_BASE_URL:", API_BASE_URL);
console.log("   NODE_ENV:", process.env.NODE_ENV);
console.log("----------------------------------------");

export default API_BASE_URL;
