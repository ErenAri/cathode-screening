import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CathodeScreen - AI-Powered Battery Materials Discovery",
  description: "Free, open-source tool for predicting cathode material stability using machine learning",
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
