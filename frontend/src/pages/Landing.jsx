import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { auth, db, provider } from '../firebase'
import { signInWithPopup } from 'firebase/auth'
import { doc, setDoc, serverTimestamp } from 'firebase/firestore'

function Landing() {
  const navigate = useNavigate()
  const [testimonialIndex, setTestimonialIndex] = useState(0)
  const [openFaq, setOpenFaq] = useState(null)

  async function handleGoogleSignIn() {
    try {
      const res = await signInWithPopup(auth, provider)
      const user = res.user
      if (user?.uid) {
        await setDoc(doc(db, 'users', user.uid), {
          uid: user.uid,
          email: user.email || '',
          displayName: user.displayName || '',
          photoURL: user.photoURL || '',
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp()
        }, { merge: true })
        navigate('/chat')
      }
    } catch (e) {
      console.error('Sign-in error', e)
    }
  }

  const testimonials = [
    {
      quote: 'Helped our team validate groundwater estimates in minutes instead of hours.',
      author: 'Ishan, Hydrologist'
    },
    {
      quote: 'The chatbot distilled complex reports into actionable guidance for our district.',
      author: 'Priya, Planning Officer'
    },
    {
      quote: 'Simple sign-in, clear answers, and great reliability for field work.',
      author: 'Amit, Field Engineer'
    }
  ]

  useEffect(() => {
    const id = setInterval(() => {
      setTestimonialIndex(prev => (prev + 1) % testimonials.length)
    }, 4000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="landing-root">
      {/* Header */}
      <header className="landing-header glass fade-in-up">
        <div className="landing-header-left">
          <div className="logo-circle">💧</div>
          <span className="brand">Ground Water Companion</span>
        </div>
        <nav className="landing-nav">
          <Link to="/">Home</Link>
          <a href="#features">Features</a>
          <button onClick={handleGoogleSignIn}>Login</button>
          <button onClick={handleGoogleSignIn}>Sign up</button>
        </nav>
      </header>

      {/* Hero */}
      <section className="landing-hero">
        <div className="hero-grid glass fade-in-up" style={{
          background: 'var(--gradient-surface)',
          border: '1px solid var(--color-border)',
          boxShadow: 'var(--shadow-2xl)',
          borderRadius: 24
        }}>
          <div className="hero-copy">
            <h1 className="landing-title">Smarter Groundwater Insights</h1>
            <p className="landing-sub">Ask questions, analyze datasets, and get guidance for groundwater estimation.</p>
            <div className="hero-ctas">
              <button onClick={handleGoogleSignIn} className="btn-primary">Sign in with Google</button>
            </div>
            <div className="hero-stats">
              <div className="stat" style={{
                background: 'var(--gradient-primary)',
                color: 'white',
                border: 'none',
                boxShadow: 'var(--shadow-lg), 0 0 0 1px rgba(255, 255, 255, 0.2)',
                borderRadius: 16
              }}>
                <div className="stat-value">10k+</div>
                <div className="stat-label">Queries answered</div>
              </div>
              <div className="stat" style={{
                background: 'var(--gradient-secondary)',
                color: 'white',
                border: 'none',
                boxShadow: 'var(--shadow-lg), 0 0 0 1px rgba(255, 255, 255, 0.2)',
                borderRadius: 16
              }}>
                <div className="stat-value">99.9%</div>
                <div className="stat-label">Uptime</div>
              </div>
              <div className="stat" style={{
                background: 'var(--gradient-accent)',
                color: 'white',
                border: 'none',
                boxShadow: 'var(--shadow-lg), 0 0 0 1px rgba(255, 255, 255, 0.2)',
                borderRadius: 16
              }}>
                <div className="stat-value">AI</div>
                <div className="stat-label">Powered analysis</div>
              </div>
            </div>
          </div>
          <div className="hero-visual float-y">
            <img alt="Groundwater illustration" className="hero-img" src="https://images.unsplash.com/photo-1644368846443-f7560dde6222?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" />
          </div>
        </div>
      </section>

      {/* Overview / About */}
      <section className="landing-about glass">
        <div className="about-grid">
          <div className="about-card about-ingres">
            <h2 style={{ marginTop: 0 }}>About INGRES</h2>
            <p>
              The Assessment of Dynamic Ground Water Resources of India is conducted annually by CGWB and State/UT Ground Water Departments under the coordination of CLEG, DoWR, RD & GR, MoJS. Using the GIS-based web app INGRES (developed by CGWB and IIT Hyderabad), the process estimates annual recharge, extractable resources, total extraction, and stage of extraction for each assessment unit. Units are categorized as Safe, Semi-Critical, Critical, or Over-Exploited.
            </p>
            <p>
              Explore the official portal: <a href="https://ingres.iith.ac.in/home" target="_blank" rel="noreferrer">ingres.iith.ac.in</a>
            </p>
            <p>
              Assessments are computed at the Block/Mandal/Taluk level and aggregated for districts and states, forming the scientific basis for groundwater planning, regulation, and conservation.
            </p>
            <h4 style={{ marginBottom: '0.5rem' }}>Key outputs</h4>
            <ul>
              <li>Annual groundwater recharge</li>
              <li>Extractable groundwater resources</li>
              <li>Total groundwater extraction</li>
              <li>Stage of groundwater extraction (Safe → Over-Exploited)</li>
            </ul>
          </div>
          <div className="about-card about-points">
            <h3 style={{ marginTop: 0 }}>Why an AI Chatbot?</h3>
            <ul>
              <li>🤖 Intelligent query handling for groundwater datasets</li>
              <li>⏱️ Real-time access to current and historical assessments</li>
              <li>📈 Interactive diagrams and visualizations</li>
              <li>🗣️ Multilingual support for Indian regional languages</li>
              <li>🔗 Seamless integration with the INGRES dataset</li>
            </ul>
            <h3>Impact</h3>
            <ul>
              <li>🧭 Easier retrieval of vast datasets for all users</li>
              <li>🧠 Informed decision-making for planners and policymakers</li>
              <li>📚 Better accessibility for researchers and the public</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Segments (inspired by reference) */}
      <section className="landing-segments">
        <div className="segments-grid">
          <div className="segment fade-in-up">
            <div className="segment-icon-circle">
              <span aria-hidden>💧</span>
            </div>
            <h3 className="segment-title">About Groundwater</h3>
            <p className="segment-desc">We all rely on groundwater in some way. Get concise, friendly explanations and key facts that align with INGRES outputs.</p>
            <button type="button" className="btn-primary" onClick={() => navigate('/groundwater')}>Learn more</button>
          </div>
          <div className="segment fade-in-up" style={{ animationDelay: '120ms' }}>
            <div className="segment-icon-circle">
              <span aria-hidden>🎓</span>
            </div>
            <h3 className="segment-title">For Students & Educators</h3>
            <p className="segment-desc">Discover ready-to-use prompts, examples, and visuals to bring groundwater estimation concepts to life.</p>
            <button type="button" onClick={() => navigate('/resources')} className="segment-btn-solid">Activities and more</button>
          </div>
        </div>
      </section>

      {/* Removed features cards as requested */}

      {/* How it works */}
      <section id="features" className="landing-how glass">
        <h2>How it works</h2>
        <div className="how-steps">
          <div className="how-step">
            <img alt="Sign up" src="https://www.shutterstock.com/image-illustration/linear-simple-black-sign-button-260nw-1791428420.jpg" />
            <h4>1. Sign in</h4>
            <p>Use your Google account to get started in seconds.</p>
          </div>
          <div className="how-step">
            <img alt="Ask" src="https://www.shutterstock.com/image-vector/chatbotchat-ai-digital-chat-bot-600nw-2277764989.jpg" />
            <h4>2. Ask</h4>
            <p>Type your groundwater questions and reference your datasets.</p>
          </div>
          <div className="how-step">
            <img alt="Act" src="https://www.shutterstock.com/image-photo/advertising-product-photo-want-create-600nw-2592005549.jpg" />
            <h4>3. Act</h4>
            <p>ChatBOT provides concise answers and insights to make decisions quickly.</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer glass">
        <div className="footer-left">
          <div className="logo-circle small">💧</div>
          <span className="brand small">Ground Water Companion</span>
        </div>
        <div className="footer-right">
          <Link to="/chat">Get started</Link>
          <a href="#features">Features</a>
          <a href="mailto:support@example.com">Support</a>
        </div>
      </footer>

      {/* Floating Chatbot Button (frontend only) */}
      <button type="button" className="fab-chat" aria-label="Open chatbot">
        <svg className="fab-icon" width="28" height="28" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden>
          <path fill="white" d="M4 6.75C4 4.679 5.679 3 7.75 3h8.5C18.321 3 20 4.679 20 6.75v5.5A3.75 3.75 0 0 1 16.25 16H11l-3.75 3v-3H7.75A3.75 3.75 0 0 1 4 12.25v-5.5Zm4.25 2a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm3.75 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm3.75 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Z"/>
        </svg>
      </button>
    </div>
  )
}

export default Landing


