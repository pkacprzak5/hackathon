# cogniro

Quiz app for WI

This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

### Prerequisites

- **Node.js 24** — With nvm: `nvm use` (run `nvm install` if needed)
- **pnpm 10+** — `corepack enable` then `corepack prepare pnpm@latest --activate`
- **GitHub CLI** — optional, for skills; `brew install gh` then `gh auth login`
- **CodeRabbit CLI** — optional, for skills; `brew install coderabbit` then `coderabbit auth login`

### Quick commands

| Command           | Description                                      |
| ----------------- | ------------------------------------------------ |
| `pnpm dev`        | Run the local development server.                |
| `pnpm build`      | Create an optimized production build.            |
| `pnpm start`      | Start the production server from the build.      |
| `pnpm lint`       | Run ESLint and fail on any warnings or errors.   |
| `pnpm lint:fix`   | Run ESLint and automatically fix fixable issues. |
| `pnpm type-check` | Run Next typegen and TypeScript type checking.   |
| `pnpm validate`   | Run lint and type-check together.                |

First, install packages:

```bash
pnpm install
```

Second, run the development server

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.