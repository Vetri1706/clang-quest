import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {
  title: 'Questline — Your C++ practice lab',
  description:
    'Learn C++ through real-world missions, a built-in editor, and a local study mentor.',
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
