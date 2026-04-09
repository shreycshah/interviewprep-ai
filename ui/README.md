# InterviewPrep AI — Frontend

A Next.js chat application that provides a conversational interface for asking questions about real interview experiences. Powered by a RAG (Retrieval-Augmented Generation) backend that retrieves relevant interview reports and generates grounded answers with source citations.

## Project Structure

```
ui/
  app/
    layout.tsx                  # Root layout with Header and Footer
    page.tsx                    # Chat page with message list and input
  components/
    chat/
      ChatInput.tsx             # Auto-resizing textarea with send button
      MessageBubble.tsx         # User/assistant message rendering with markdown and sources
      MessageList.tsx           # Scrollable message list with suggestion prompts
      SourceCard.tsx            # Source attribution card for retrieved chunks
    layout/
      Header.tsx                # Site header with logo and "Chat" badge
      Footer.tsx                # Site footer with project tagline
  lib/
    api.ts                      # Typed API client (sendMessage -> POST /api/chat)
    types.ts                    # TypeScript interfaces (ChatMessage, ChatResponse, ChatSource)
```

## Features

- **Chat interface** — Send interview prep questions and receive AI-generated answers grounded in real interview experiences from Google, Meta, Amazon, and more
- **Source citations** — Assistant responses include clickable links to the original interview reports used to generate the answer
- **Markdown rendering** — Bold text, bullet lists, and inline URLs are rendered in assistant messages
- **Suggestion prompts** — First-time users see example questions to get started
- **Loading indicators** — Animated dot pulse while waiting for responses
- **Latency display** — Response time shown on each assistant message
- **Auto-scroll** — Message list scrolls to the latest message automatically
- **Keyboard submit** — Press Enter to send, Shift+Enter for newline

## Local Setup

1. Install dependencies:
   ```
   cd ui
   npm install
   ```

2. Create `.env.local` from the example:
   ```
   cp .env.local.example .env.local
   ```

3. Set the backend URL in `.env.local`:
   ```
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   ```

4. Start the dev server:
   ```
   npm run dev
   ```

   The app runs at `http://localhost:3000`.

## Backend Connection

The frontend communicates with the FastAPI backend via a single endpoint:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Send a message and receive a RAG-generated response with sources |

All requests go through the `NEXT_PUBLIC_API_BASE_URL` environment variable. The backend must be running with the RAG pipeline initialized for the chat to work.
