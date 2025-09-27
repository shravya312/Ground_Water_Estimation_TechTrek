/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Custom color palette based on existing CSS variables
                primary: {
                    50: '#eff6ff',
                    100: '#dbeafe',
                    200: '#bfdbfe',
                    300: '#93c5fd',
                    400: '#60a5fa',
                    500: '#3b82f6',
                    600: '#2563eb',
                    700: '#1d4ed8',
                    800: '#1e40af',
                    900: '#1e3a8a',
                },
                secondary: {
                    50: '#ecfeff',
                    100: '#cffafe',
                    200: '#a5f3fc',
                    300: '#67e8f9',
                    400: '#22d3ee',
                    500: '#06b6d4',
                    600: '#0891b2',
                    700: '#0e7490',
                    800: '#155e75',
                    900: '#164e63',
                },
                accent: {
                    50: '#faf5ff',
                    100: '#f3e8ff',
                    200: '#e9d5ff',
                    300: '#d8b4fe',
                    400: '#c084fc',
                    500: '#a855f7',
                    600: '#9333ea',
                    700: '#7c3aed',
                    800: '#6b21a8',
                    900: '#581c87',
                },
                // Dark theme colors
                dark: {
                    background: '#0f172a',
                    'background-secondary': '#1e293b',
                    surface: '#1e293b',
                    'surface-elevated': '#334155',
                    'surface-hover': '#475569',
                    'text-primary': '#ffffff',
                    'text-secondary': '#e2e8f0',
                    'text-muted': '#94a3b8',
                    border: '#334155',
                    'border-light': '#475569',
                    'border-dark': '#1e293b',
                },
                // Navy palette
                navy: {
                    50: '#f8fafc',
                    100: '#f1f5f9',
                    200: '#e2e8f0',
                    300: '#cbd5e1',
                    400: '#94a3b8',
                    500: '#64748b',
                    600: '#475569',
                    700: '#334155',
                    800: '#1e293b',
                    900: '#0f172a',
                },
                // Success, warning, error colors
                success: '#10b981',
                warning: '#f59e0b',
                error: '#ef4444',
            },
            fontFamily: {
                sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
            },
            animation: {
                'float-up': 'floatUp 600ms cubic-bezier(0.4, 0, 0.2, 1) forwards',
                'slide-in-right': 'slideInRight 500ms cubic-bezier(0.4, 0, 0.2, 1) forwards',
                'slide-in-left': 'slideInLeft 500ms cubic-bezier(0.4, 0, 0.2, 1) forwards',
                'float-y': 'floatY 4.5s ease-in-out infinite',
                'pulse-glow': 'pulse-glow 1.5s ease-out infinite',
                'focus-pulse': 'focusPulse 1s ease',
            },
            keyframes: {
                floatUp: {
                    from: {
                        opacity: '0',
                        transform: 'translateY(20px)',
                    },
                    to: {
                        opacity: '1',
                        transform: 'translateY(0)',
                    },
                },
                slideInRight: {
                    from: {
                        opacity: '0',
                        transform: 'translateX(30px)',
                    },
                    to: {
                        opacity: '1',
                        transform: 'translateX(0)',
                    },
                },
                slideInLeft: {
                    from: {
                        opacity: '0',
                        transform: 'translateX(-30px)',
                    },
                    to: {
                        opacity: '1',
                        transform: 'translateX(0)',
                    },
                },
                floatY: {
                    '0%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-6px)' },
                    '100%': { transform: 'translateY(0)' },
                },
                'pulse-glow': {
                    '0%': { boxShadow: '0 0 0 0 rgba(220,38,38,0.6)' },
                    '70%': { boxShadow: '0 0 0 10px rgba(220,38,38,0)' },
                    '100%': { boxShadow: '0 0 0 0 rgba(220,38,38,0)' },
                },
                focusPulse: {
                    '0%': { boxShadow: '0 0 0 0 rgba(153, 176, 176, 0.35)' },
                    '100%': { boxShadow: '0 0 0 8px rgba(153, 176, 176, 0)' },
                },
            },
            backdropBlur: {
                xs: '2px',
            },
            boxShadow: {
                'glass': '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(59, 130, 246, 0.2)',
                'glass-hover': '0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(59, 130, 246, 0.4)',
            },
            backgroundImage: {
                'gradient-primary': 'linear-gradient(135deg, var(--tw-gradient-stops))',
                'gradient-secondary': 'linear-gradient(135deg, #06b6d4, #22d3ee)',
                'gradient-accent': 'linear-gradient(135deg, #8b5cf6, #a78bfa)',
                'gradient-surface': 'linear-gradient(135deg, #1e293b, #334155)',
                'gradient-background': 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)',
                'gradient-card': 'linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.6))',
            },
        },
    },
    plugins: [],
}