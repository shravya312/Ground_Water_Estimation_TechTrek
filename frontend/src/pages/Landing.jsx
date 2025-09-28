import { useEffect, useState, useRef } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { auth, db, provider } from '../firebase'
import { signInWithPopup } from 'firebase/auth'
import { doc, setDoc, serverTimestamp } from 'firebase/firestore'
import '../index.css'

function Landing() {
  const navigate = useNavigate()
  const [testimonialIndex, setTestimonialIndex] = useState(0)
  const [openFaq, setOpenFaq] = useState(null)
  const [visibleElements, setVisibleElements] = useState(new Set())
  const [isScrolled, setIsScrolled] = useState(false)
  const elementRefs = useRef({})

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

  // Scroll animation effect
  useEffect(() => {
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setVisibleElements(prev => new Set([...prev, entry.target.id]))
        }
      })
    }, observerOptions)

    // Observe all elements with data-animate attribute
    const animateElements = document.querySelectorAll('[data-animate]')
    animateElements.forEach((el) => {
      observer.observe(el)
    })

    return () => {
      observer.disconnect()
    }
  }, [])

  // Scroll event listener for header behavior
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop
      setIsScrolled(scrollTop > 50) // Trigger separation after 50px scroll
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
      <div style={{ 
        minHeight: '100vh', 
        display: 'flex', 
        flexDirection: 'column', 
        backgroundColor: 'white',
        fontFamily: '"Lato", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
      }}>
      <style>
        {`
          #hero-description,
          #hero-title {
            color: white !important;
          }
          .landing-hero {
            color: white !important;
          }
          .hero-section {
            color: white !important;
          }
        `}
      </style>

      {/* Header */}
      <header style={{
        background: 'linear-gradient(90deg, rgba(255, 255, 255, 0.9) 48%, rgba(240, 249, 255, 0.9) 52%)',
        backdropFilter: 'blur(15px)',
        WebkitBackdropFilter: 'blur(15px)',
        color: '#333',
        padding: '0.5rem 1rem',
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        borderBottom: isScrolled ? '1px solid rgba(59, 130, 246, 0.2)' : 'none',
        boxShadow: isScrolled ? '0 4px 20px rgba(59, 130, 246, 0.1)' : 'none',
        transition: 'all 0.3s ease'
      }}>
        <div style={{ 
          margin: '0',
          paddingLeft: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          {/* Logo */}
          <div style={{ 
            flexShrink: 0,
            backgroundColor: '#1e3a8a',
            padding: '0.25rem',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '45px',
            height: '45px'
          }}>
            <img 
              src="/logo1.png" 
              alt="Central Ground Water Board Logo" 
              style={{
                height: '30px',
                width: 'auto',
                objectFit: 'contain'
              }}
            />
          </div>
          
          {/* Text Content */}
          <div style={{ textAlign: 'left' }}>
            <h1 style={{ 
              fontSize: '0.95rem',
              fontWeight: 'bold',
              margin: '0 0 0.125rem 0',
              color: '#333',
              letterSpacing: '0.1px'
            }}>
              Central Ground Water Board
            </h1>
            <p style={{ 
              fontSize: '0.65rem',
              margin: '0 0 0.0625rem 0',
              color: '#666',
              fontWeight: '500'
            }}>
              Department of WR, RD & GR
            </p>
            <p style={{ 
              fontSize: '0.55rem',
              margin: '0',
              color: '#888',
              fontWeight: '400'
            }}>
              Ministry of Jal Shakti, Government of India
            </p>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem 1rem',
        marginTop: isScrolled ? '100px' : '80px',
        transition: 'margin-top 0.3s ease'
      }}>
        <div style={{
          width: '100%',
          maxWidth: window.innerWidth > 1200 ? '1200px' : window.innerWidth > 768 ? '1000px' : '95%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          padding: window.innerWidth > 1024 ? '4rem 4rem' : window.innerWidth > 768 ? '3.5rem 3rem' : '3rem 2rem',
          backgroundImage: 'url("/bg.png")',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          border: '1px solid #e5e7eb',
          borderRadius: window.innerWidth > 768 ? '32px' : '24px',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.1)',
          animation: 'floatUp 0.6s ease-out',
          position: 'relative',
          minHeight: window.innerWidth > 1024 ? '450px' : window.innerWidth > 768 ? '400px' : '350px'
        }}>
          {/* Background overlay for better text readability */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.4)',
            borderRadius: window.innerWidth > 768 ? '32px' : '24px',
            zIndex: 1
          }}></div>
            <h1 id="hero-title" className="landing-hero hero-section" style={{
              fontSize: window.innerWidth > 1200 ? '3rem' : window.innerWidth > 1024 ? '2.75rem' : window.innerWidth > 768 ? '2.5rem' : window.innerWidth > 640 ? '2.25rem' : '2rem',
            fontWeight: 'bold',
              color: 'white',
            lineHeight: 1.1,
            margin: '0 0 1.5rem 0',
            position: 'relative',
              zIndex: 2,
              textShadow: '2px 2px 4px rgba(0, 0, 0, 0.7)'
          }}>
            Ground Water Companion
          </h1>
          <p id="hero-description" className="landing-hero hero-section" style={{
            fontSize: window.innerWidth > 1200 ? '1.5rem' : window.innerWidth > 1024 ? '1.375rem' : window.innerWidth > 768 ? '1.25rem' : window.innerWidth > 640 ? '1.125rem' : '1rem',
            color: 'white !important',
            lineHeight: 1.5,
            margin: '0 0 2.5rem 0',
            maxWidth: window.innerWidth > 768 ? '900px' : '700px',
            position: 'relative',
            zIndex: 2,
            textShadow: '1px 1px 3px rgba(0, 0, 0, 0.7)'
          }}>
            Your intelligent assistant for groundwater estimation, analysis, and conservation. Get instant insights, expert guidance, and actionable recommendations.
          </p>
          <button 
            onClick={handleGoogleSignIn} 
            style={{
              padding: '16px 40px',
              backgroundColor: '#0ea5e9',
              fontWeight: '600',
              borderRadius: '50px',
              border: 'none',
              cursor: 'pointer',
              fontSize: '1rem',
              transition: 'all 0.3s ease',
              boxShadow: '0 4px 15px 0 rgba(14, 165, 233, 0.3)',
              position: 'relative',
              zIndex: 2,
              color: '#1e40af',
              fontFamily: 'sans-serif',
              letterSpacing: '0.5px'
            }}
            onMouseOver={(e) => {
              e.target.style.backgroundColor = '#38bdf8';
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = '0 6px 20px 0 rgba(14, 165, 233, 0.4)';
            }}
            onMouseOut={(e) => {
              e.target.style.backgroundColor = '#0ea5e9';
              e.target.style.transform = 'translateY(0)';
              e.target.style.boxShadow = '0 4px 15px 0 rgba(14, 165, 233, 0.3)';
            }}
          >
            Sign in with Google
          </button>
        </div>
      </section>

      {/* About Section */}
      <section style={{
        padding: '6rem 2rem',
        backgroundColor: 'white',
        position: 'relative'
      }}>
        
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          position: 'relative',
          zIndex: 2
        }}>
          {/* Header */}
          <div 
            className={`animate-fade-in-up ${visibleElements.has('about-header') ? 'visible' : ''}`}
            id="about-header"
            data-animate
            style={{ textAlign: 'center', marginBottom: '4rem' }}
          >
          <h2 style={{ 
              fontSize: '3rem',
              fontWeight: 'bold',
              color: '#0ea5e9',
              margin: '0 0 1rem 0',
              lineHeight: 1.1
            }}>
              About INGRES Platform
            </h2>
            <div style={{
              width: '80px',
              height: '4px',
              backgroundColor: '#0ea5e9',
              margin: '0 auto',
              borderRadius: '2px'
            }}></div>
          </div>
          
          {/* Main Content */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem'
          }}>
            {/* INGRES Platform Card */}
            <div 
              className={`animate-fade-in-up ${visibleElements.has('ingres-card') ? 'visible' : ''}`}
              id="ingres-card"
              data-animate
              style={{
                background: 'white',
                borderRadius: '20px',
                padding: '3rem',
                boxShadow: '0 10px 30px rgba(14, 165, 233, 0.1)',
                border: '2px solid #e0f2fe',
                transform: 'translateY(0)',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => {
                e.target.style.transform = 'translateY(-10px)';
                e.target.style.boxShadow = '0 20px 40px rgba(14, 165, 233, 0.2)';
                e.target.style.borderColor = '#0ea5e9';
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 10px 30px rgba(14, 165, 233, 0.1)';
                e.target.style.borderColor = '#e0f2fe';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2rem' }}>
                <div style={{
                  width: '60px',
                  height: '60px',
                  background: 'linear-gradient(135deg, #0ea5e9, #1e40af)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginRight: '1.5rem',
                  fontSize: '1.5rem'
                }}>
                  🌐
                </div>
                <div>
                  <h3 style={{
                    fontSize: '2rem',
                    fontWeight: 'bold',
                    color: '#1e293b',
                    margin: '0 0 0.5rem 0'
                  }}>
                    INGRES GIS Platform
                  </h3>
                  <p style={{
                    fontSize: '1rem',
                    color: '#64748b',
                    margin: '0'
                  }}>
                    Indian Groundwater Resource Estimation System
                  </p>
                </div>
              </div>
              
              <p style={{
                fontSize: '1.125rem',
                color: '#475569',
                lineHeight: 1.7,
                margin: '0 0 2rem 0'
              }}>
                The INGRES platform provides comprehensive groundwater data visualization and analysis tools. 
                Our chatbot integrates with this powerful system to deliver real-time insights and accurate 
                groundwater estimations across India.
              </p>
              
              <a 
                href="https://ingres.iith.ac.in/gecdataonline/gis/INDIA;parentLocName=INDIA;locname=INDIA;loctype=COUNTRY;view=ADMIN;locuuid=ffce954d-24e1-494b-ba7e-0931d8ad6085;year=2024-2025;computationType=normal;component=recharge;period=annual;category=safe;mapOnClickParams=false"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '1rem 2rem',
                  background: 'linear-gradient(135deg, #0ea5e9, #1e40af)',
                  color: 'white',
                  textDecoration: 'none',
                  borderRadius: '50px',
                  fontWeight: '600',
                  fontSize: '1rem',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 15px rgba(14, 165, 233, 0.4)'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-2px)';
                  e.target.style.boxShadow = '0 8px 25px rgba(14, 165, 233, 0.6)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.boxShadow = '0 4px 15px rgba(14, 165, 233, 0.4)';
                }}
              >
                Explore INGRES Platform
                <span style={{ fontSize: '1.2rem' }}>→</span>
              </a>
            </div>

            {/* AI Features Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '2rem'
            }}>
              <div 
                className={`animate-fade-in-left ${visibleElements.has('feature-1') ? 'visible' : ''}`}
                id="feature-1"
                data-animate
                style={{
                  background: 'white',
                  borderRadius: '16px',
                  padding: '2rem',
                  textAlign: 'center',
                  border: '2px solid #e0f2fe',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 15px rgba(14, 165, 233, 0.1)'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-5px)';
                  e.target.style.borderColor = '#0ea5e9';
                  e.target.style.boxShadow = '0 8px 25px rgba(14, 165, 233, 0.15)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.borderColor = '#e0f2fe';
                  e.target.style.boxShadow = '0 4px 15px rgba(14, 165, 233, 0.1)';
                }}
              >
                <div style={{
                  width: '50px',
                  height: '50px',
                  background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1rem',
                  fontSize: '1.5rem'
                }}>
                  🤖
                </div>
                <h4 style={{
                  fontSize: '1.25rem',
                  fontWeight: 'bold',
                  color: '#1e293b',
                  margin: '0 0 1rem 0'
                }}>
                  AI-Powered Analysis
                </h4>
                <p style={{
                  fontSize: '0.95rem',
                  color: '#64748b',
                  lineHeight: 1.6,
                  margin: '0'
                }}>
                  Advanced algorithms provide instant groundwater estimations and expert-level insights.
                </p>
              </div>

              <div 
                className={`animate-fade-in-up ${visibleElements.has('feature-2') ? 'visible' : ''}`}
                id="feature-2"
                data-animate
                style={{
                  background: 'white',
                  borderRadius: '16px',
                  padding: '2rem',
                  textAlign: 'center',
                  border: '2px solid #e0f2fe',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 15px rgba(14, 165, 233, 0.1)'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-5px)';
                  e.target.style.borderColor = '#0ea5e9';
                  e.target.style.boxShadow = '0 8px 25px rgba(14, 165, 233, 0.15)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.borderColor = '#e0f2fe';
                  e.target.style.boxShadow = '0 4px 15px rgba(14, 165, 233, 0.1)';
                }}
              >
                <div style={{
                  width: '50px',
                  height: '50px',
                  background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1rem',
                  fontSize: '1.5rem'
                }}>
                  📊
                </div>
                <h4 style={{
                  fontSize: '1.25rem',
                  fontWeight: 'bold',
                  color: '#1e293b',
                  margin: '0 0 1rem 0'
                }}>
                  Real-time Data
                </h4>
                <p style={{
                  fontSize: '0.95rem',
                  color: '#64748b',
                  lineHeight: 1.6,
                  margin: '0'
                }}>
                  Access live groundwater data and comprehensive resource estimation across India.
                </p>
              </div>

              <div 
                className={`animate-fade-in-right ${visibleElements.has('feature-3') ? 'visible' : ''}`}
                id="feature-3"
                data-animate
                style={{
                  background: 'white',
                  borderRadius: '16px',
                  padding: '2rem',
                  textAlign: 'center',
                  border: '2px solid #e0f2fe',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 4px 15px rgba(14, 165, 233, 0.1)'
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = 'translateY(-5px)';
                  e.target.style.borderColor = '#0ea5e9';
                  e.target.style.boxShadow = '0 8px 25px rgba(14, 165, 233, 0.15)';
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = 'translateY(0)';
                  e.target.style.borderColor = '#e0f2fe';
                  e.target.style.boxShadow = '0 4px 15px rgba(14, 165, 233, 0.1)';
                }}
              >
                <div style={{
                  width: '50px',
                  height: '50px',
                  background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 1rem',
                  fontSize: '1.5rem'
                }}>
                  🏛️
                </div>
                <h4 style={{
                  fontSize: '1.25rem',
            fontWeight: 'bold',
                  color: '#1e293b',
                  margin: '0 0 1rem 0'
                }}>
                  Official CGWB Data
                </h4>
                <p style={{
                  fontSize: '0.95rem',
                  color: '#64748b',
                  lineHeight: 1.6,
                  margin: '0'
                }}>
                  Powered by official Central Ground Water Board data and IIT Hyderabad research.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer 
        className={`animate-fade-in-up ${visibleElements.has('footer') ? 'visible' : ''}`}
        id="footer"
        data-animate
        style={{
          backgroundColor: 'white',
          color: '#333',
          padding: '2rem 1.5rem',
          boxShadow: '0 -2px 10px rgba(0, 0, 0, 0.1)',
          borderTop: '2px solid #e0f2fe'
        }}
      >
        <div style={{ 
          maxWidth: '1200px', 
          margin: '0 auto',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '2rem',
          alignItems: 'flex-start'
        }}>
          {/* Left Side - Logo and Title */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '1rem',
            minWidth: '300px',
            flex: '1'
          }}>
            {/* Logo */}
            <div style={{
              backgroundColor: '#1e3a8a',
              padding: '0.5rem',
              borderRadius: '50%',
              width: '50px',
              height: '50px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0
            }}>
              <img
                src="/logo1.png"
                alt="Central Ground Water Board Logo"
                style={{
                  height: '30px',
                  width: 'auto',
                  objectFit: 'contain'
                }}
              />
            </div>

            {/* Title */}
            <div style={{ flex: 1 }}>
              <h2 style={{ 
                fontSize: window.innerWidth > 768 ? '1.25rem' : '1rem',
                fontWeight: 'bold',
                margin: '0 0 0.25rem 0',
                color: '#0ea5e9'
          }}>
            Central Ground Water Board
          </h2>
          <p style={{ 
            fontSize: window.innerWidth > 768 ? '0.875rem' : '0.75rem',
                margin: '0 0 0.125rem 0',
                color: '#666'
          }}>
            Department of WR, RD & GR
          </p>
          <p style={{ 
                fontSize: window.innerWidth > 768 ? '0.75rem' : '0.65rem',
                margin: '0',
                color: '#888'
              }}>
                Ministry of Jal Shakti, Government of India
              </p>
            </div>
          </div>

          {/* Middle - Contributors */}
          <div style={{
            minWidth: '250px',
            flex: '1',
            textAlign: 'center'
          }}>
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 'bold',
              color: '#0ea5e9',
              margin: '0 0 1rem 0'
            }}>
              Development Team
            </h3>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.25rem'
            }}>
              <p style={{
                fontSize: '0.875rem',
                margin: '0',
                color: '#333',
                fontWeight: '500'
              }}>
                Shravya H Jain
              </p>
              <p style={{
                fontSize: '0.875rem',
                margin: '0',
                color: '#333',
                fontWeight: '500'
              }}>
                Srishyla Kumar TP
              </p>
              <p style={{
                fontSize: '0.875rem',
                margin: '0',
                color: '#333',
                fontWeight: '500'
              }}>
                Rakshita RL
              </p>
              <p style={{
                fontSize: '0.875rem',
                margin: '0',
                color: '#333',
                fontWeight: '500'
              }}>
                Shreesha S Shetty
              </p>
              <p style={{
                fontSize: '0.875rem',
                margin: '0',
                color: '#333',
                fontWeight: '500'
              }}>
                Mayuri J Shetty
              </p>
              <p style={{
                fontSize: '0.875rem',
            margin: '0',
                color: '#333',
                fontWeight: '500'
              }}>
                Mohan R
              </p>
            </div>
          </div>

          {/* Right Side - Contact Us */}
          <div style={{
            minWidth: '200px',
            flex: '1',
            textAlign: 'right'
          }}>
            <h3 style={{
              fontSize: '1rem',
              fontWeight: 'bold',
              color: '#0ea5e9',
              margin: '0 0 1rem 0'
            }}>
              Contact Us
            </h3>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: '0.5rem'
              }}>
                <span style={{ fontSize: '1rem' }}>📧</span>
                <a 
                  href="mailto:info@cgwb.gov.in" 
                  style={{
                    fontSize: '0.875rem',
                    color: '#333',
                    textDecoration: 'none'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#0ea5e9'}
                  onMouseOut={(e) => e.target.style.color = '#333'}
                >
                  info@cgwb.gov.in
                </a>
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: '0.5rem'
              }}>
                <span style={{ fontSize: '1rem' }}>🌐</span>
                <a 
                  href="https://cgwb.gov.in" 
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: '0.875rem',
                    color: '#333',
                    textDecoration: 'none'
                  }}
                  onMouseOver={(e) => e.target.style.color = '#0ea5e9'}
                  onMouseOut={(e) => e.target.style.color = '#333'}
                >
                  cgwb.gov.in
                </a>
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: '0.5rem'
              }}>
                <span style={{ fontSize: '1rem' }}>📍</span>
                <span style={{
                  fontSize: '0.875rem',
                  color: '#666'
                }}>
                  New Delhi, India
                </span>
              </div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: '0.5rem'
              }}>
                <span style={{ fontSize: '1rem' }}>☎️</span>
                <span style={{
                  fontSize: '0.875rem',
                  color: '#666'
                }}>
                  +91-11-2616 5277
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Border */}
        <div style={{
          marginTop: '1.5rem',
          paddingTop: '1rem',
          borderTop: '1px solid #e0f2fe',
          textAlign: 'center'
        }}>
          <p style={{
            fontSize: '0.75rem',
            color: '#888',
            margin: '0'
          }}>
            © 2024 Central Ground Water Board. All rights reserved.
          </p>
        </div>
      </footer>

    </div>
  )
}

export default Landing


