/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      colors: {
        navy: {
          850: "#131c31",
          900: "#0f172a",
          950: "#0a0f1e",
        },
        brand: {
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
      },
      boxShadow: {
        glow: "0 0 40px -8px rgba(6, 182, 212, 0.35)",
        "glow-sm": "0 4px 24px -4px rgba(37, 99, 235, 0.25)",
        glass: "0 8px 32px rgba(15, 23, 42, 0.08)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "glow-pulse": {
          "0%, 100%": { opacity: "0.5", transform: "scale(1)" },
          "50%": { opacity: "0.85", transform: "scale(1.05)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.45" },
        },
        "typing-bounce": {
          "0%, 60%, 100%": { transform: "translateY(0)" },
          "30%": { transform: "translateY(-4px)" },
        },
        "mic-pulse": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(6, 182, 212, 0.35)" },
          "50%": { boxShadow: "0 0 0 6px rgba(6, 182, 212, 0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.35s ease-out forwards",
        "glow-pulse": "glow-pulse 3s ease-in-out infinite",
        "pulse-soft": "pulse-soft 2s ease-in-out infinite",
        "typing-bounce": "typing-bounce 1.2s ease-in-out infinite",
        "mic-pulse": "mic-pulse 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
