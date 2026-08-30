import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CourtSignal",
  description:
    "Basketball player impact in points per 100 possessions: SPM, RAPM, AIO, and roles.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

/**
 * Stamps the saved theme before first paint so the page never flashes the wrong
 * surface, and so the toggle can read the current theme straight off the DOM.
 */
const themeScript = `try{var t=localStorage.getItem("impact-theme")||"dark";if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){document.documentElement.dataset.theme="dark"}`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
