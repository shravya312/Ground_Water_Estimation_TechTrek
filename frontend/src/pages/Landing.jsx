import { useEffect, useState, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { auth, db, provider } from '../firebase'
import { signInWithPopup, onAuthStateChanged } from 'firebase/auth'
import { doc, setDoc, serverTimestamp } from 'firebase/firestore'

function Landing() {
  const navigate = useNavigate()
  const [testimonialIndex, setTestimonialIndex] = useState(0)
  const [openFaq, setOpenFaq] = useState(null)
  const [isScrolled, setIsScrolled] = useState(false)
  const [isFooterScrolled, setIsFooterScrolled] = useState(false)
  const [position, setPosition] = useState({ 
    x: window.innerWidth - 100, 
    y: Math.max(60, window.innerHeight - 100) // Ensure it starts below navbar
  })
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [hasMoved, setHasMoved] = useState(false)
  const [dragStartPosition, setDragStartPosition] = useState({ x: 0, y: 0 })
  const [user, setUser] = useState(null)
  const [showSignInPrompt, setShowSignInPrompt] = useState(false)
  const chatbotRef = useRef(null)

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
      }
    } catch (e) {
      console.error('Sign-in error', e)
    }
  }

  const handleMouseDown = (e) => {
    if (e.button !== 0) return // Only left mouse button
    setIsDragging(true)
    setHasMoved(false)
    const rect = chatbotRef.current.getBoundingClientRect()
    setDragStartPosition({ x: rect.left, y: rect.top })
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    })
    e.preventDefault()
  }

  const handleMouseMove = (e) => {
    if (!isDragging) return
    
    const newX = e.clientX - dragOffset.x
    const newY = e.clientY - dragOffset.y
    
    // Check if the button has actually moved from its starting position
    const currentPosition = { x: newX, y: newY }
    const distance = Math.sqrt(
      Math.pow(currentPosition.x - dragStartPosition.x, 2) + 
      Math.pow(currentPosition.y - dragStartPosition.y, 2)
    )
    
    if (distance > 5) { // Threshold of 5 pixels to consider it a movement
      setHasMoved(true)
    }
    
    // Keep within screen bounds but below the navbar (60px header height)
    const maxX = window.innerWidth - 60
    const maxY = window.innerHeight - 60
    const minY = 60 // Below the navbar
    
    setPosition({
      x: Math.max(0, Math.min(newX, maxX)),
      y: Math.max(minY, Math.min(newY, maxY))
    })
  }

  const handleMouseUp = (e) => {
    setIsDragging(false)
    
    // Calculate the current position based on mouse position
    const currentX = e.clientX - dragOffset.x
    const screenWidth = window.innerWidth
    const iconWidth = 60
    
    // Keep within bounds first
    const maxX = screenWidth - iconWidth
    const boundedX = Math.max(0, Math.min(currentX, maxX))
    
    // Determine which side is closer
    const distanceToLeft = boundedX
    const distanceToRight = screenWidth - boundedX - iconWidth
    
    let newX
    if (distanceToLeft < distanceToRight) {
      // Snap to left side
      newX = 20 // 20px from left edge
    } else {
      // Snap to right side
      newX = screenWidth - iconWidth - 20 // 20px from right edge
    }
    
    setPosition(prev => ({
      ...prev,
      x: newX
    }))
    
    // Reset hasMoved after a short delay to allow clicking after drag
    setTimeout(() => {
      setHasMoved(false)
    }, 100)
  }

  const handleClick = (e) => {
    if (isDragging || hasMoved) {
      e.preventDefault()
      return
    }
    if (user) {
      navigate('/chat')
    } else {
      setShowSignInPrompt(true)
      setTimeout(() => setShowSignInPrompt(false), 3000)
    }
  }

  const handleWaterDropletClick = () => {
    if (user) {
      navigate('/chat')
    } else {
      setShowSignInPrompt(true)
      setTimeout(() => setShowSignInPrompt(false), 3000)
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

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY
      const windowHeight = window.innerHeight
      const documentHeight = document.documentElement.scrollHeight
      
      setIsScrolled(scrollTop > 50)
      
      // Check if we're near the footer (within 200px of bottom)
      const distanceFromBottom = documentHeight - (scrollTop + windowHeight)
      setIsFooterScrolled(distanceFromBottom < 200)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user)
    })
    return () => unsubscribe()
  }, [])

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.userSelect = 'none'
      document.body.style.cursor = 'grabbing'
    } else {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.userSelect = ''
      document.body.style.cursor = ''
    }
  }, [isDragging, dragOffset])

  return (
    <div className="landing-root" style={{
      background: 'white'
    }}>
      {/* Header */}
      <header className="landing-header" style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        background: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        padding: '0.5rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '60px',
        transition: 'all 0.3s ease'
      }}>
        <div className="header-left" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="cgwb-logo" style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            background: '#1e40af', 
            width: '45px',
            height: '45px',
            borderRadius: '50%',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}>
            <img 
              src="/images/logo1-Picsart-AiImageEnhancer[1].png" 
              alt="Central Ground Water Board Logo" 
              style={{ width: '35px', height: '35px', objectFit: 'contain' }}
            />
          </div>
          <div className="cgwb-text">
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold', color: '#0f172a' }}>Central Ground Water Board</div>
            <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Department of WR, RD & GR</div>
            <div style={{ fontSize: '0.6rem', color: '#94a3b8' }}>Ministry of Jal Shakti, Government of India</div>
          </div>
        </div>
        <nav className="header-nav" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <a href="#features" style={{ 
            color: 'white', 
            textDecoration: 'none', 
            fontSize: '0.9rem',
            background: '#1e40af',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            fontWeight: '500',
            transition: 'background-color 0.2s ease'
          }}>Features</a>
          <a href="#about" style={{ 
            color: 'white', 
            textDecoration: 'none', 
            fontSize: '0.9rem',
            background: '#1e40af',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            fontWeight: '500',
            transition: 'background-color 0.2s ease'
          }}>About</a>
          <div className="nav-icons" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button className="nav-icon-btn" aria-label="Data Input" style={{
              background: '#1e40af',
              border: 'none',
              padding: '0.5rem',
              cursor: 'pointer',
              borderRadius: '6px',
              transition: 'background-color 0.2s ease'
            }}>
              <img 
                src="/images/input-icon.svg" 
                alt="Data Input Icon" 
                style={{ width: '20px', height: '20px', filter: 'brightness(0) invert(1)' }}
              />
            </button>
            <div style={{ width: '1px', height: '20px', background: '#e2e8f0' }}></div>
            <button className="nav-icon-btn" aria-label="Language Selection" style={{
              background: '#1e40af',
              border: 'none',
              padding: '0.5rem',
              cursor: 'pointer',
              borderRadius: '6px',
              transition: 'background-color 0.2s ease'
            }}>
              <img 
                src="/images/multilingual-icon.svg" 
                alt="Language Selection Icon" 
                style={{ width: '20px', height: '20px', filter: 'brightness(0) invert(1)' }}
              />
            </button>
          </div>
        </nav>
      </header>

      {/* Hero */}
      <section className="landing-hero" style={{ 
        marginTop: '60px', 
        height: '60vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundImage: "url('https://images.unsplash.com/photo-1644368846443-f7560dde6222?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')",
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        width: '100%'
      }}>
        <div className="hero-grid" style={{
          padding: '2rem',
          position: 'relative',
          width: '100%',
          maxWidth: '800px',
          textAlign: 'center'
        }}>
          {user ? (
            <>
              <h1 className="landing-title" style={{ 
                color: 'white', 
                textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)',
                fontSize: '2rem',
                marginBottom: '0.5rem'
              }}>Welcome, {user.displayName || user.email?.split('@')[0] || 'User'}!</h1>
              <p className="landing-sub" style={{ 
                color: 'white', 
                textShadow: '1px 1px 2px rgba(0, 0, 0, 0.5)',
                fontSize: '1.2rem',
                marginBottom: '1rem'
              }}>Ready to explore groundwater insights? Click the water droplet to start chatting!</p>
              <div className="hero-ctas" style={{ marginBottom: '1rem' }}>
                <button onClick={() => navigate('/chat')} style={{
                  background: '#1e3a8a',
                  color: 'white',
                  border: 'none',
                  padding: '0.6rem 1.2rem',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)'
                }}>Go to Chat</button>
              </div>
            </>
          ) : (
            <>
              <h1 className="landing-title" style={{ 
                color: 'white', 
                textShadow: '2px 2px 4px rgba(0, 0, 0, 0.5)',
                fontSize: '2rem',
                marginBottom: '0.5rem'
              }}>Smarter Groundwater Insights</h1>
              <p className="landing-sub" style={{ 
                color: 'white', 
                textShadow: '1px 1px 2px rgba(0, 0, 0, 0.5)',
                fontSize: '1rem',
                marginBottom: '1rem'
              }}>Ask questions, analyze datasets, and get guidance for groundwater estimation.</p>
              <div className="hero-ctas" style={{ marginBottom: '1rem' }}>
                <button onClick={handleGoogleSignIn} style={{
                  background: '#1e3a8a',
                  color: 'white',
                  border: 'none',
                  padding: '0.6rem 1.2rem',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)'
                }}>Sign in with Google</button>
              </div>
            </>
          )}
          <div className="hero-stats" style={{ 
            display: 'flex', 
            gap: '1rem', 
            justifyContent: 'center',
            flexWrap: 'wrap'
          }}>
            <div className="stat">
              <div className="stat-value" style={{ 
                color: 'white', 
                fontSize: '1.2rem', 
                fontWeight: 'bold', 
                textShadow: '1px 1px 2px rgba(0, 0, 0, 0.5)' 
              }}>10k+</div>
              <div className="stat-label" style={{ 
                color: 'rgba(255, 255, 255, 0.9)', 
                fontSize: '0.8rem' 
              }}>Queries</div>
            </div>
            <div className="stat">
              <div className="stat-value" style={{ 
                color: 'white', 
                fontSize: '1.2rem', 
                fontWeight: 'bold', 
                textShadow: '1px 1px 2px rgba(0, 0, 0, 0.5)' 
              }}>99.9%</div>
              <div className="stat-label" style={{ 
                color: 'rgba(255, 255, 255, 0.9)', 
                fontSize: '0.8rem' 
              }}>Uptime</div>
            </div>
            <div className="stat">
              <div className="stat-value" style={{ 
                color: 'white', 
                fontSize: '1.2rem', 
                fontWeight: 'bold', 
                textShadow: '1px 1px 2px rgba(0, 0, 0, 0.5)' 
              }}>AI</div>
              <div className="stat-label" style={{ 
                color: 'rgba(255, 255, 255, 0.9)', 
                fontSize: '0.8rem' 
              }}>Powered</div>
            </div>
          </div>
        </div>
      </section>

      {/* Overview / About */}
      <section className="landing-about">
        <div className="about-grid">
                <div className="about-card about-ingres">
            <h2>About INGRES (INDIA-Groundwater Resource Estimation System)</h2>
            <p>
              The Assessment of Dynamic Ground Water Resources of India is conducted annually by CGWB and State/UT Ground Water Departments. Using the GIS-based web app INGRES (INDIA-Groundwater Resource Estimation System) developed by CGWB and IIT Hyderabad, the process estimates annual recharge, extractable resources, total extraction, and stage of extraction for each assessment unit.
            </p>
            <p>
              Explore the official portal: <a href="https://ingres.iith.ac.in/home" target="_blank" rel="noreferrer">ingres.iith.ac.in</a>
            </p>
            <h4>Key outputs</h4>
            <ul>
              <li>Annual groundwater recharge</li>
              <li>Extractable groundwater resources</li>
              <li>Total groundwater extraction</li>
              <li>Stage of groundwater extraction (Safe → Over-Exploited)</li>
            </ul>
          </div>
          <div className="about-card about-points">
            <h3>Why an AI Chatbot?</h3>
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

     

      {/* How it works */}
      <section id="features" className="landing-how">
        <h2>How it works</h2>
        <div className="timeline-container">
          <div className="timeline-line"></div>
          
          <div className="timeline-step step-left">
            <div className="step-content">
              <div className="step-image-circle">
                <img 
                  alt="Sign up" 
                  src="https://www.shutterstock.com/image-illustration/linear-simple-black-sign-button-260nw-1791428420.jpg" 
                />
              </div>
              <div className="step-text">
                <h4>1. Sign in</h4>
                <p>Use your Google account to get started in seconds.</p>
              </div>
            </div>
          </div>

          <div className="timeline-step step-right">
            <div className="step-content">
              <div className="step-image-circle">
                <img 
                  alt="Ask" 
                  src="https://www.shutterstock.com/image-vector/chatbotchat-ai-digital-chat-bot-600nw-2277764989.jpg" 
                />
              </div>
              <div className="step-text">
                <h4>2. Ask</h4>
                <p>Type your groundwater questions and reference your datasets.</p>
              </div>
            </div>
          </div>

          <div className="timeline-step step-left">
            <div className="step-content">
              <div className="step-image-circle">
                <img 
                  alt="Act" 
                  src="https://www.shutterstock.com/image-photo/advertising-product-photo-want-create-600nw-2592005549.jpg" 
                />
              </div>
              <div className="step-text">
                <h4>3. Act</h4>
                <p>ChatBOT provides concise answers and insights to make decisions quickly.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        background: '#1e3a8a',
        color: 'white',
        padding: '3rem 2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: '2rem'
      }}>
        {/* Left side - Logo and header info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flex: '1' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            background: 'rgba(255, 255, 255, 0.2)', 
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}>
            <img 
              src="/images/logo1-Picsart-AiImageEnhancer[1].png" 
              alt="Central Ground Water Board Logo" 
              style={{ width: '45px', height: '45px', objectFit: 'contain' }}
            />
          </div>
          <div>
            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Central Ground Water Board</div>
            <div style={{ fontSize: '1rem', marginBottom: '0.25rem' }}>Department of WR, RD & GR</div>
            <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Ministry of Jal Shakti, Government of India</div>
          </div>
        </div>

        {/* Middle - Contributors */}
        <div style={{ flex: '1', textAlign: 'center' }}>
          <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem' }}>Contributors</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
            <div>Central Ground Water Board (CGWB)</div>
            <div>IIT Hyderabad</div>
            <div>Ministry of Jal Shakti</div>
            <div>State/UT Ground Water Departments</div>
          </div>
        </div>

        {/* Right side - Social icons and links */}
        <div style={{ flex: '1', textAlign: 'right' }}>
          <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem' }}>Connect With Us</h3>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem', marginBottom: '1rem' }}>
            <a href="#" style={{ color: 'white', fontSize: '1.5rem' }}>📧</a>
            <a href="#" style={{ color: 'white', fontSize: '1.5rem' }}>🐦</a>
            <a href="#" style={{ color: 'white', fontSize: '1.5rem' }}>📘</a>
            <a href="#" style={{ color: 'white', fontSize: '1.5rem' }}>🔗</a>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.9rem' }}>
            <Link to="/chat" style={{ color: 'white', textDecoration: 'none' }}>Get Started</Link>
            <a href="#features" style={{ color: 'white', textDecoration: 'none' }}>Features</a>
            <a href="mailto:support@example.com" style={{ color: 'white', textDecoration: 'none' }}>Support</a>
          </div>
        </div>
      </footer>

      {/* Floating Chatbot Button (frontend only) */}
      <button 
        ref={chatbotRef}
        type="button" 
        className="fab-chat"
        aria-label="Open chatbot"
        onMouseDown={handleMouseDown}
        onClick={handleWaterDropletClick}
        style={{
          position: 'fixed',
          left: `${position.x}px`,
          top: `${position.y}px`,
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          background: '#1e3a8a',
          border: 'none',
          cursor: isDragging ? 'grabbing' : 'grab',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: isDragging ? 'none' : 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          zIndex: 1000,
          userSelect: 'none'
        }}
        onMouseEnter={(e) => {
          if (!isDragging) {
            e.target.style.transform = 'scale(1.1)'
          }
        }}
        onMouseLeave={(e) => {
          if (!isDragging) {
            e.target.style.transform = 'scale(1)'
          }
        }}
      >
        <img 
          src="/images/droplet.jpg" 
          alt="Water Drop Chatbot Icon" 
          style={{ width: '60px', height: '60px', objectFit: 'cover', borderRadius: '50%', pointerEvents: 'none' }}
        />
      </button>

      {/* Sign-in Prompt */}
      {showSignInPrompt && (
        <div style={{
          position: 'fixed',
          bottom: '80px',
          right: '2rem',
          background: '#1e3a8a',
          color: 'white',
          padding: '1rem',
          borderRadius: '8px',
          fontSize: '0.9rem',
          fontWeight: '600',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1001,
          animation: 'fadeInOut 3s ease-in-out',
          maxWidth: '200px'
        }}>
          Please sign in first to access the chat
          <div style={{
            position: 'absolute',
            bottom: '-6px',
            right: '20px',
            width: '0',
            height: '0',
            borderLeft: '6px solid transparent',
            borderRight: '6px solid transparent',
            borderTop: '6px solid #1e3a8a'
          }}></div>
        </div>
      )}
    </div>
  )
}

export default Landing