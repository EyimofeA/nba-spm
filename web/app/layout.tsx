import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NBA Impact Lab",
  description:
    "Player impact in points per 100 possessions: SPM, RAPM, AIO, roles, and aging.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
};

/**
 * Stamps the saved theme before first paint so the page never flashes the wrong
 * surface, and so the toggle can read the current theme straight off the DOM.
 */
const themeScript = `try{var t=localStorage.getItem("impact-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
