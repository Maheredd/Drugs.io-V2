/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#1f77b4',
        success: '#00cc96',
        warning: '#ffa500',
        danger: '#ff4b4b',
        brand: {
          50: '#f0f8ff',
          100: '#dbe9fe',
          200: '#a8c7fa',
          300: '#7aa5f0',
          400: '#5a8fd8',
          500: '#3b7ac0',
          600: '#2d5f99',
          700: '#1f4372',
          800: '#12284b',
          900: '#050e24',
        }
      },
      animation: {
        'spin': 'spin 1s linear infinite',
      }
    },
  },
  plugins: [],
}
