"use client";

import Link from "next/link";
import API_BASE_URL from "@/config";
import { SocialLinks } from "@/components/SocialLinks";

export default function Home() {
  return (
    <div className="min-h-screen bg-white font-sans text-gray-900">
      {/* Header */}
      <header className="fixed w-full top-4 z-50 flex justify-center pointer-events-none">
        <div className="bg-white/80 backdrop-blur-md border border-white/50 shadow-lg rounded-full px-8 py-3 flex items-center gap-12 pointer-events-auto">
          <Link href="/" className="font-bold text-gray-900 text-lg tracking-tight hover:text-blue-600 hover:scale-125 transition-all duration-200">
            CathodeScreen
          </Link>
          <nav className="flex items-center gap-8 text-sm font-medium text-gray-600">
            <Link href="/predict" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Predict</Link>
            <Link href="/database" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Database</Link>
            <Link href="/database#screening-pack" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">Screening</Link>
            <Link href="/about" className="hover:text-blue-600 hover:scale-125 transition-all duration-200">About</Link>
          </nav>
          <div className="pl-8 border-l border-gray-200">
            <a href={`${API_BASE_URL}/docs`} target="_blank" className="text-xs font-bold text-blue-600 hover:text-blue-700 hover:scale-125 transition-all duration-200 uppercase tracking-wide flex items-center gap-1">
              API <span className="text-[10px]">↗</span>
            </a>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 max-w-6xl mx-auto text-center">
        <div className="inline-block px-4 py-1.5 bg-blue-50 text-blue-700 rounded-full text-sm font-medium mb-6">
          Enterprise AI Screening
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight mb-6 text-gray-900">
          Accelerate Battery Material <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
            Discovery with AI
          </span>
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-10 leading-relaxed">
          Instantly predict thermodynamic stability of Li-cathode materials using our governance-approved
          MACE-MP-0 ensemble. 92.7% KEEP precision, 0% false-kill rate, 90% calibrated coverage.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            href="/predict"
            className="px-8 py-4 bg-blue-600 text-white text-lg font-semibold rounded-xl shadow-lg shadow-blue-200 hover:bg-blue-700 hover:-translate-y-0.5 transition-all"
          >
            Try Prediction Tool →
          </Link>
          <Link
            href="/database#screening-pack"
            className="px-8 py-4 bg-emerald-600 text-white text-lg font-semibold rounded-xl shadow-lg shadow-emerald-200 hover:bg-emerald-700 hover:-translate-y-0.5 transition-all"
          >
            Open Screening Pack
          </Link>
          <Link
            href="/database"
            className="px-8 py-4 bg-white text-gray-700 text-lg font-semibold rounded-xl border border-gray-200 shadow-sm hover:bg-gray-50 hover:-translate-y-0.5 transition-all"
          >
            Browse Database
          </Link>
        </div>

        {/* How It Works */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto mt-20 text-left">
          <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mb-4 text-blue-600 font-bold">1</div>
            <h3 className="font-semibold text-lg mb-2">Upload Structure</h3>
            <p className="text-gray-600 text-sm">Drag & drop your CIF file. Our model parses 3D crystal structures instantly.</p>
          </div>
          <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center mb-4 text-indigo-600 font-bold">2</div>
            <h3 className="font-semibold text-lg mb-2">MACE Ensemble Inference</h3>
            <p className="text-gray-600 text-sm">5-member MACE-MP-0 fine-tuned ensemble predicts E<sub>hull</sub> with conformal calibration.</p>
          </div>
          <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center mb-4 text-green-600 font-bold">3</div>
            <h3 className="font-semibold text-lg mb-2">Get Recommendation</h3>
            <p className="text-gray-600 text-sm">Receive KEEP, MAYBE, or KILL recommendations to prioritize your research.</p>
          </div>
        </div>
      </section>

      {/* Results / Methodology Section */}
      <section className="py-20 px-6 max-w-6xl mx-auto bg-gray-50 rounded-3xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-6">Built on Rigorous Science</h2>
            <p className="text-lg text-gray-600 mb-6">
              We leverage a 5-member deep ensemble of MACE-MP-0 (Multi-Atomic Cluster Expansion) foundation models, fine-tuned on 17,227 Li-cathode materials from the Materials Project.
            </p>
            <ul className="space-y-4 mb-8">
              <li className="flex items-start gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-xs font-bold">✓</div>
                <div>
                  <h4 className="font-semibold text-gray-900">SOAP-LOCO Validation</h4>
                  <p className="text-sm text-gray-600">Validated on clustered hold-out sets to ensure generalization to new chemistries.</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-xs font-bold">✓</div>
                <div>
                  <h4 className="font-semibold text-gray-900">Uncertainty Quantification</h4>
                  <p className="text-sm text-gray-600">Aleatoric (quantile regression) + epistemic (inter-model disagreement) + conformal calibration for 90% coverage.</p>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-green-100 flex items-center justify-center text-green-600 text-xs font-bold">✓</div>
                <div>
                  <h4 className="font-semibold text-gray-900">Decision-Grade Output</h4>
                  <p className="text-sm text-gray-600">Clear KEEP/MAYBE/KILL recommendations for immediate lab prioritization.</p>
                </div>
              </li>
            </ul>
            <Link href="/about" className="text-blue-600 font-semibold hover:text-blue-700 flex items-center gap-2 group">
              Read full methodology
              <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" /></svg>
            </Link>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 text-center">
              <p className="text-4xl font-extrabold text-blue-600 mb-2">2.12×</p>
              <p className="font-semibold text-gray-900">Enrichment Factor</p>
              <p className="text-xs text-gray-500 mt-2">EF@10 on test set</p>
            </div>
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 text-center">
              <p className="text-4xl font-extrabold text-green-600 mb-2">0.030</p>
              <p className="font-semibold text-gray-900">MAE (eV/atom)</p>
              <p className="text-xs text-gray-500 mt-2">Test set (1,013 materials)</p>
            </div>
            <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 text-center">
              <p className="text-4xl font-extrabold text-indigo-600 mb-2">0.0%</p>
              <p className="font-semibold text-gray-900">False Kill Rate</p>
              <p className="text-xs text-gray-500 mt-2">No stable materials discarded</p>
            </div>
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-2xl text-white text-center">
              <p className="text-3xl font-bold mb-2">17,227</p>
              <p className="text-gray-400">TMO Cathode Materials</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200 py-12 mt-20">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-gray-500 mb-4">CathodeScreen · Enterprise Platform · 2026</p>
          <div className="flex justify-center mb-6">
            <SocialLinks />
          </div>
          <div className="flex justify-center gap-6 text-sm text-gray-600 mb-4">
            <Link href="/about" className="hover:text-blue-600">About</Link>
            <Link href="/predict" className="hover:text-blue-600">Predict</Link>
            <Link href="/database" className="hover:text-blue-600">Database</Link>
            <Link href="/database#screening-pack" className="hover:text-blue-600">Screening</Link>
          </div>
          <p className="text-xs text-gray-400 border-t border-gray-200 pt-4">
            ⚠️ For screening purposes only. DFT validation recommended before synthesis. Model optimized for Li-oxide cathodes.
          </p>
        </div>
      </footer>
    </div>
  );
}
