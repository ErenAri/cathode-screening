import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CathodeScreen | Battery Material Discovery",
  description: "AI-powered platform for predicting cathode material stability using machine learning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
