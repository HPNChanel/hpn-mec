import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'

export default [
  { ignores: ['dist'] },
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...js.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
  {
    files: ['**/*.tsx'],
    rules: {
      'no-restricted-syntax': ['error', 'TSXElement', 'TSXFragment'],
      'no-duplicate-imports': 'error',
      'no-restricted-extensions': [
        'error',
        {
          "extensions": [".tsx"]
        }
      ]
    }
  },
  {
    files: ['**/*.ts'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: "ExportNamedDeclaration:not([exportKind='type'])",
          message: "Only type exports are allowed in .ts files."
        },
        {
          selector: "ImportDeclaration:not([importKind='type'])",
          message: "Only type imports are allowed in .ts files."
        }
      ]
    }
  }
]
