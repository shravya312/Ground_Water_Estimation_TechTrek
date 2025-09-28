import { useEffect, useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { auth, db, provider } from "../firebase";
import { signInWithPopup, onAuthStateChanged } from "firebase/auth";
import { doc, setDoc, serverTimestamp } from "firebase/firestore";
import "../index.css";

function Landing() {
  const navigate = useNavigate();
  const [visibleElements, setVisibleElements] = useState(new Set());
  const [isScrolled, setIsScrolled] = useState(false);
  const [user, setUser] = useState(null);
  const [buttonPosition, setButtonPosition] = useState({
    x: window.innerWidth - 80,
    y: window.innerHeight - 80,
  }); // Default position bottom right
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [hasMoved, setHasMoved] = useState(false);
  const [dragStartPosition, setDragStartPosition] = useState({ x: 0, y: 0 });
  const [justFinishedDragging, setJustFinishedDragging] = useState(false);
  const [isClick, setIsClick] = useState(true);
  const elementRefs = useRef({});
  const buttonRef = useRef(null);

  async function handleGoogleSignIn() {
    try {
      const res = await signInWithPopup(auth, provider);
      const user = res.user;
      if (user?.uid) {
        await setDoc(
          doc(db, "users", user.uid),
          {
            uid: user.uid,
            email: user.email || "",
            displayName: user.displayName || "",
            photoURL: user.photoURL || "",
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
          },
          { merge: true }
        );
        navigate("/chat");
      }
    } catch (e) {
      console.error("Sign-in error", e);
    }
  }

  // Scroll animation effect
  useEffect(() => {
    const observerOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -50px 0px",
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setVisibleElements((prev) => new Set([...prev, entry.target.id]));
        }
      });
    }, observerOptions);

    // Observe all elements with data-animate attribute
    const animateElements = document.querySelectorAll("[data-animate]");
    animateElements.forEach((el) => {
      observer.observe(el);
    });

    return () => {
      observer.disconnect();
    };
  }, []);

  // Scroll event listener for header behavior
  useEffect(() => {
    const handleScroll = () => {
      const scrollTop =
        window.pageYOffset || document.documentElement.scrollTop;
      setIsScrolled(scrollTop > 50); // Trigger separation after 50px scroll
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Check authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
    });

    return () => unsubscribe();
  }, []);

  // Handle chatbot button click
  const handleChatbotClick = async () => {
    console.log("handleChatbotClick called - user:", user);
    try {
      if (user) {
        // User is signed in, navigate to chat
        console.log("User is signed in, navigating to chat");
        navigate("/chat");
      } else {
        // User is not signed in, trigger sign-in
        console.log("User not signed in, triggering sign-in");
        await handleGoogleSignIn();
      }
    } catch (error) {
      console.error("Error in handleChatbotClick:", error);
    }
  };

  // Calculate which side is closer and snap to it
  const snapToClosestSide = (x) => {
    const windowWidth = window.innerWidth;
    const buttonWidth = 60; // Button width

    // Check if closer to left or right edge
    const distanceToLeft = x;
    const distanceToRight = windowWidth - (x + buttonWidth);

    if (distanceToLeft < distanceToRight) {
      return 20; // Snap to left side
    } else {
      return windowWidth - buttonWidth - 20; // Snap to right side
    }
  };

  // Handle mouse down for dragging
  const handleMouseDown = (e) => {
    e.preventDefault();
    const rect = buttonRef.current.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;

    setDragOffset({ x: offsetX, y: offsetY });
    setDragStartPosition({ x: e.clientX, y: e.clientY });
    setHasMoved(false);
    setIsDragging(true);
    setIsClick(true); // Reset click state
  };

  // Handle mouse move for dragging
  const handleMouseMove = (e) => {
    if (!isDragging) return;

    // Check if mouse has moved significantly (more than 5 pixels)
    const deltaX = Math.abs(e.clientX - dragStartPosition.x);
    const deltaY = Math.abs(e.clientY - dragStartPosition.y);

    if (deltaX > 5 || deltaY > 5) {
      setHasMoved(true);
      setIsClick(false); // Mark as not a click when moved significantly
    }

    const newX = e.clientX - dragOffset.x;
    const newY = e.clientY - dragOffset.y;

    // Keep button within viewport bounds
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const buttonWidth = 60;
    const buttonHeight = 60;
    const headerHeight = 0; // No header, so no height constraint

    const constrainedX = Math.max(0, Math.min(newX, windowWidth - buttonWidth));
    const constrainedY = Math.max(
      headerHeight,
      Math.min(newY, windowHeight - buttonHeight)
    );

    setButtonPosition({ x: constrainedX, y: constrainedY });
  };

  // Handle mouse up - snap to closest side
  const handleMouseUp = () => {
    if (!isDragging) return;

    // Only snap if the button was actually moved
    if (hasMoved) {
      const snappedX = snapToClosestSide(buttonPosition.x);
      setButtonPosition((prev) => ({ ...prev, x: snappedX }));
    }

    setIsDragging(false);
    setHasMoved(false);
    setJustFinishedDragging(true);

    // Reset the flag after a short delay to allow for future clicks
    setTimeout(() => {
      setJustFinishedDragging(false);
      setIsClick(true); // Reset click state
    }, 100);
  };

  // Add event listeners for dragging
  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);

      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, dragOffset, buttonPosition]);

  // Handle touch events for mobile
  const handleTouchStart = (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = buttonRef.current.getBoundingClientRect();
    const offsetX = touch.clientX - rect.left;
    const offsetY = touch.clientY - rect.top;

    setDragOffset({ x: offsetX, y: offsetY });
    setDragStartPosition({ x: touch.clientX, y: touch.clientY });
    setHasMoved(false);
    setIsDragging(true);
    setIsClick(true); // Reset click state
  };

  const handleTouchMove = (e) => {
    if (!isDragging) return;

    e.preventDefault();
    const touch = e.touches[0];

    // Check if touch has moved significantly (more than 5 pixels)
    const deltaX = Math.abs(touch.clientX - dragStartPosition.x);
    const deltaY = Math.abs(touch.clientY - dragStartPosition.y);

    if (deltaX > 5 || deltaY > 5) {
      setHasMoved(true);
      setIsClick(false); // Mark as not a click when moved significantly
    }

    const newX = touch.clientX - dragOffset.x;
    const newY = touch.clientY - dragOffset.y;

    // Keep button within viewport bounds
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const buttonWidth = 60;
    const buttonHeight = 60;
    const headerHeight = 0; // No header, so no height constraint

    const constrainedX = Math.max(0, Math.min(newX, windowWidth - buttonWidth));
    const constrainedY = Math.max(
      headerHeight,
      Math.min(newY, windowHeight - buttonHeight)
    );

    setButtonPosition({ x: constrainedX, y: constrainedY });
  };

  const handleTouchEnd = () => {
    if (!isDragging) return;

    // Only snap if the button was actually moved
    if (hasMoved) {
      const snappedX = snapToClosestSide(buttonPosition.x);
      setButtonPosition((prev) => ({ ...prev, x: snappedX }));
    }

    setIsDragging(false);
    setHasMoved(false);
    setJustFinishedDragging(true);

    // Reset the flag after a short delay to allow for future clicks
    setTimeout(() => {
      setJustFinishedDragging(false);
      setIsClick(true); // Reset click state
    }, 100);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        backgroundColor: "white",
        fontFamily:
          '"Lato", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
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

      {/* Header - Commented out for now, can be restored later */}
      {/*
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
      */}

      {/* Navbar */}
      <nav
        style={{
          background:
            "linear-gradient(90deg, rgba(255, 255, 255, 0.95) 48%, rgba(240, 249, 255, 0.95) 52%)",
          backdropFilter: "blur(15px)",
          WebkitBackdropFilter: "blur(15px)",
          color: "#333",
          padding: "0.75rem 1.5rem",
          position: "sticky",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          borderBottom: "1px solid rgba(59, 130, 246, 0.1)",
          boxShadow: "0 2px 10px rgba(0, 0, 0, 0.1)",
          transition: "all 0.3s ease",
        }}
      >
        <div
          style={{
            maxWidth: "1200px",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
          }}
        >
          {/* Logo and Title Section */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
            }}
          >
            {/* Logo */}
            <div
              style={{
                flexShrink: 0,
                backgroundColor: "#1e3a8a",
                padding: "0.5rem",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: "40px",
                height: "40px",
              }}
            >
              <img
                src="/logo1.png"
                alt="Central Ground Water Board Logo"
                style={{
                  height: "28px",
                  width: "auto",
                  objectFit: "contain",
                }}
              />
            </div>

            {/* Text Content */}
            <div style={{ textAlign: "left" }}>
              <h1
                style={{
                  fontSize: "0.95rem",
                  fontWeight: "bold",
                  margin: "0 0 0.25rem 0",
                  color: "#333",
                  letterSpacing: "0.1px",
                }}
              >
                Central Ground Water Board
              </h1>
              <p
                style={{
                  fontSize: "0.65rem",
                  margin: "0 0 0.125rem 0",
                  color: "#666",
                  fontWeight: "500",
                }}
              >
                Department of WR, RD & GR
              </p>
              <p
                style={{
                  fontSize: "0.55rem",
                  margin: "0",
                  color: "#888",
                  fontWeight: "400",
                }}
              >
                Ministry of Jal Shakti, Government of India
              </p>
            </div>
          </div>

          {/* Profile Section */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
            }}
          >
            {/* INGRES Platform Link */}
            <a
              href="https://ingres.iith.ac.in/gecdataonline/gis/INDIA;parentLocName=INDIA;locname=INDIA;loctype=COUNTRY;view=ADMIN;locuuid=ffce954d-24e1-494b-ba7e-0931d8ad6085;year=2024-2025;computationType=normal;component=recharge;period=annual;category=safe;mapOnClickParams=false"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: "#333",
                textDecoration: "none",
                fontSize: "0.9rem",
                fontWeight: "500",
                padding: "0.5rem 1rem",
                borderRadius: "8px",
                transition: "all 0.3s ease",
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = "#0ea5e9";
                e.target.style.color = "white";
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = "transparent";
                e.target.style.color = "#333";
              }}
            >
              INGRES Platform
            </a>

            {/* Profile Icon */}
            {user ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    padding: "0.5rem",
                    borderRadius: "50px",
                    backgroundColor: "rgba(14, 165, 233, 0.1)",
                    border: "2px solid rgba(14, 165, 233, 0.2)",
                    transition: "all 0.3s ease",
                    cursor: "pointer",
                  }}
                  onMouseOver={(e) => {
                    e.target.style.backgroundColor = "rgba(14, 165, 233, 0.2)";
                    e.target.style.borderColor = "rgba(14, 165, 233, 0.4)";
                  }}
                  onMouseOut={(e) => {
                    e.target.style.backgroundColor = "rgba(14, 165, 233, 0.1)";
                    e.target.style.borderColor = "rgba(14, 165, 233, 0.2)";
                  }}
                >
                  {user.photoURL ? (
                    <img
                      src={user.photoURL}
                      alt="User Profile"
                      style={{
                        width: "30px",
                        height: "30px",
                        borderRadius: "50%",
                        objectFit: "cover",
                        border: "2px solid white",
                      }}
                    />
                  ) : (
                    <div
                      style={{
                        width: "30px",
                        height: "30px",
                        borderRadius: "50%",
                        backgroundColor: "#0ea5e9",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "1rem",
                        color: "white",
                        border: "2px solid white",
                      }}
                    >
                      👤
                    </div>
                  )}
                  <span
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: "500",
                      color: "#333",
                      marginRight: "0.5rem",
                    }}
                  >
                    {user.displayName || user.email?.split("@")[0] || "User"}
                  </span>
                </div>

                {/* Logout Button */}
                <button
                  onClick={async () => {
                    try {
                      await auth.signOut();
                      console.log("User signed out successfully");
                    } catch (error) {
                      console.error("Error signing out:", error);
                    }
                  }}
                  style={{
                    width: "32px",
                    height: "32px",
                    borderRadius: "8px",
                    backgroundColor: "rgba(107, 114, 128, 0.1)",
                    border: "1px solid rgba(107, 114, 128, 0.2)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "pointer",
                    transition: "all 0.3s ease",
                    padding: "0",
                    overflow: "hidden",
                  }}
                  onMouseOver={(e) => {
                    e.target.style.backgroundColor = "rgba(239, 68, 68, 0.1)";
                    e.target.style.borderColor = "rgba(239, 68, 68, 0.3)";
                    e.target.style.color = "#dc2626";
                    e.target.style.transform = "translateY(-1px)";
                  }}
                  onMouseOut={(e) => {
                    e.target.style.backgroundColor = "rgba(107, 114, 128, 0.1)";
                    e.target.style.borderColor = "rgba(107, 114, 128, 0.2)";
                    e.target.style.color = "#6b7280";
                    e.target.style.transform = "translateY(0)";
                  }}
                  title="Sign Out"
                  aria-label="Sign Out"
                >
                  <div
                    style={{
                      width: "20px",
                      height: "20px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "1rem",
                      color: "#6b7280",
                      transition: "all 0.3s ease",
                    }}
                  >
                    ↪
                  </div>
                </button>
              </div>
            ) : (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  padding: "0.5rem",
                  borderRadius: "50px",
                  backgroundColor: "rgba(245, 158, 11, 0.1)",
                  border: "2px solid rgba(245, 158, 11, 0.2)",
                  transition: "all 0.3s ease",
                }}
              >
                <div
                  style={{
                    width: "30px",
                    height: "30px",
                    borderRadius: "50%",
                    backgroundColor: "#f59e0b",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "1rem",
                    color: "white",
                    border: "2px solid white",
                  }}
                >
                  👤
                </div>
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: "500",
                    color: "#333",
                    marginRight: "0.5rem",
                  }}
                >
                  Guest
                </span>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem 1rem",
          marginTop: "0px",
          transition: "margin-top 0.3s ease",
          backgroundImage: 'url("/bg.png")',
          backgroundSize: "cover",
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          position: "relative",
        }}
      >
        {/* Background overlay for better text readability */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundColor: "rgba(0, 0, 0, 0.4)",
            zIndex: 1,
          }}
        ></div>
        <div
          style={{
            width: "100%",
            maxWidth:
              window.innerWidth > 1200
                ? "1200px"
                : window.innerWidth > 768
                ? "1000px"
                : "95%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            textAlign: "center",
            padding:
              window.innerWidth > 1024
                ? "4rem 4rem"
                : window.innerWidth > 768
                ? "3.5rem 3rem"
                : "3rem 2rem",
            border: "1px solid rgba(255, 255, 255, 0.2)",
            borderRadius: window.innerWidth > 768 ? "32px" : "24px",
            boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.1)",
            animation: "floatUp 0.6s ease-out",
            position: "relative",
            zIndex: 2,
            minHeight:
              window.innerWidth > 1024
                ? "450px"
                : window.innerWidth > 768
                ? "400px"
                : "350px",
            backdropFilter: "blur(10px)",
            backgroundColor: "rgba(255, 255, 255, 0.05)",
          }}
        >
          <h1
            id="hero-title"
            className="landing-hero hero-section"
            style={{
              fontSize:
                window.innerWidth > 1200
                  ? "3rem"
                  : window.innerWidth > 1024
                  ? "2.75rem"
                  : window.innerWidth > 768
                  ? "2.5rem"
                  : window.innerWidth > 640
                  ? "2.25rem"
                  : "2rem",
              fontWeight: "bold",
              color: "white",
              lineHeight: 1.1,
              margin: "0 0 1.5rem 0",
              position: "relative",
              zIndex: 2,
              textShadow: "2px 2px 4px rgba(0, 0, 0, 0.7)",
            }}
          >
            Ground Water Companion
          </h1>
          <p
            id="hero-description"
            className="landing-hero hero-section"
            style={{
              fontSize:
                window.innerWidth > 1200
                  ? "1.5rem"
                  : window.innerWidth > 1024
                  ? "1.375rem"
                  : window.innerWidth > 768
                  ? "1.25rem"
                  : window.innerWidth > 640
                  ? "1.125rem"
                  : "1rem",
              color: "white !important",
              lineHeight: 1.5,
              margin: "0 0 2.5rem 0",
              maxWidth: window.innerWidth > 768 ? "900px" : "700px",
              position: "relative",
              zIndex: 2,
              textShadow: "1px 1px 3px rgba(0, 0, 0, 0.7)",
            }}
          >
            Your intelligent assistant for groundwater estimation, analysis, and
            conservation. Get instant insights, expert guidance, and actionable
            recommendations.
          </p>
          {!user && (
            <button
              onClick={handleGoogleSignIn}
              style={{
                padding: "16px 40px",
                backgroundColor: "#0ea5e9",
                fontWeight: "600",
                borderRadius: "50px",
                border: "none",
                cursor: "pointer",
                fontSize: "1rem",
                transition: "all 0.3s ease",
                boxShadow: "0 4px 15px 0 rgba(14, 165, 233, 0.3)",
                position: "relative",
                zIndex: 2,
                color: "#1e40af",
                fontFamily: "sans-serif",
                letterSpacing: "0.5px",
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = "#38bdf8";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow =
                  "0 6px 20px 0 rgba(14, 165, 233, 0.4)";
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = "#0ea5e9";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow =
                  "0 4px 15px 0 rgba(14, 165, 233, 0.3)";
              }}
            >
              Sign in with Google
            </button>
          )}

          {/* Test Chat Link */}
        </div>
      </section>

      {/* About Section */}
      <section
        id="about"
        style={{
          padding: "6rem 2rem",
          backgroundColor: "white",
          position: "relative",
        }}
      >
        <div
          style={{
            maxWidth: "1200px",
            margin: "0 auto",
            position: "relative",
            zIndex: 2,
          }}
        >
          {/* Header */}
          <div
            className={`animate-fade-in-up ${
              visibleElements.has("about-header") ? "visible" : ""
            }`}
            id="about-header"
            data-animate
            style={{ textAlign: "center", marginBottom: "4rem" }}
          >
            <h2
              style={{
                fontSize: "3rem",
                fontWeight: "bold",
                color: "#0ea5e9",
                margin: "0 0 1rem 0",
                lineHeight: 1.1,
              }}
            >
              About INGRES Platform
            </h2>
            <div
              style={{
                width: "80px",
                height: "4px",
                backgroundColor: "#0ea5e9",
                margin: "0 auto",
                borderRadius: "2px",
              }}
            ></div>
          </div>

          {/* Main Content */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "2rem",
            }}
          >
            {/* INGRES Platform Card */}
            <div
              className={`animate-fade-in-up ${
                visibleElements.has("ingres-card") ? "visible" : ""
              }`}
              id="ingres-card"
              data-animate
              style={{
                background: "white",
                borderRadius: "20px",
                padding: "3rem",
                boxShadow: "0 10px 30px rgba(14, 165, 233, 0.1)",
                border: "2px solid #e0f2fe",
                transform: "translateY(0)",
                transition: "all 0.3s ease",
              }}
              onMouseOver={(e) => {
                e.target.style.transform = "translateY(-10px)";
                e.target.style.boxShadow =
                  "0 20px 40px rgba(14, 165, 233, 0.2)";
                e.target.style.borderColor = "#0ea5e9";
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow =
                  "0 10px 30px rgba(14, 165, 233, 0.1)";
                e.target.style.borderColor = "#e0f2fe";
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  marginBottom: "2rem",
                }}
              >
                <div
                  style={{
                    width: "60px",
                    height: "60px",
                    background: "linear-gradient(135deg, #0ea5e9, #1e40af)",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginRight: "1.5rem",
                    fontSize: "1.5rem",
                  }}
                >
                  🌐
                </div>
                <div>
                  <h3
                    style={{
                      fontSize: "2rem",
                      fontWeight: "bold",
                      color: "#1e293b",
                      margin: "0 0 0.5rem 0",
                    }}
                  >
                    INGRES GIS Platform
                  </h3>
                  <p
                    style={{
                      fontSize: "1rem",
                      color: "#64748b",
                      margin: "0",
                    }}
                  >
                    Indian Groundwater Resource Estimation System
                  </p>
                </div>
              </div>

              <p
                style={{
                  fontSize: "1.125rem",
                  color: "#475569",
                  lineHeight: 1.7,
                  margin: "0 0 2rem 0",
                }}
              >
                The INGRES platform provides comprehensive groundwater data
                visualization and analysis tools. Our chatbot integrates with
                this powerful system to deliver real-time insights and accurate
                groundwater estimations across India.
              </p>

              <a
                href="https://ingres.iith.ac.in/gecdataonline/gis/INDIA;parentLocName=INDIA;locname=INDIA;loctype=COUNTRY;view=ADMIN;locuuid=ffce954d-24e1-494b-ba7e-0931d8ad6085;year=2024-2025;computationType=normal;component=recharge;period=annual;category=safe;mapOnClickParams=false"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  padding: "1rem 2rem",
                  background: "linear-gradient(135deg, #0ea5e9, #1e40af)",
                  color: "white",
                  textDecoration: "none",
                  borderRadius: "50px",
                  fontWeight: "600",
                  fontSize: "1rem",
                  transition: "all 0.3s ease",
                  boxShadow: "0 4px 15px rgba(14, 165, 233, 0.4)",
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = "translateY(-2px)";
                  e.target.style.boxShadow =
                    "0 8px 25px rgba(14, 165, 233, 0.6)";
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = "translateY(0)";
                  e.target.style.boxShadow =
                    "0 4px 15px rgba(14, 165, 233, 0.4)";
                }}
              >
                Explore INGRES Platform
                <span style={{ fontSize: "1.2rem" }}>→</span>
              </a>
            </div>

            {/* AI Features Grid */}
            <div
              id="features"
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
                gap: "2rem",
              }}
            >
              <div
                className={`animate-fade-in-left ${
                  visibleElements.has("feature-1") ? "visible" : ""
                }`}
                id="feature-1"
                data-animate
                style={{
                  background: "white",
                  borderRadius: "16px",
                  padding: "2rem",
                  textAlign: "center",
                  border: "2px solid #e0f2fe",
                  transition: "all 0.3s ease",
                  boxShadow: "0 4px 15px rgba(14, 165, 233, 0.1)",
                  transform: "translateY(0px)",
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = "translateY(-5px)";
                  e.target.style.borderColor = "#0ea5e9";
                  e.target.style.boxShadow =
                    "0 8px 25px rgba(14, 165, 233, 0.15)";
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = "translateY(0px)";
                  e.target.style.borderColor = "#e0f2fe";
                  e.target.style.boxShadow =
                    "0 4px 15px rgba(14, 165, 233, 0.1)";
                }}
              >
                <div
                  style={{
                    width: "50px",
                    height: "50px",
                    background: "linear-gradient(135deg, #0ea5e9, #0284c7)",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    margin: "0 auto 1rem",
                    fontSize: "1.5rem",
                  }}
                >
                  🤖
                </div>
                <h4
                  style={{
                    fontSize: "1.25rem",
                    fontWeight: "bold",
                    color: "#1e293b",
                    margin: "0 0 1rem 0",
                  }}
                >
                  AI-Powered Analysis
                </h4>
                <p
                  style={{
                    fontSize: "0.95rem",
                    color: "#64748b",
                    lineHeight: 1.6,
                    margin: "0",
                  }}
                >
                  Advanced algorithms provide instant groundwater estimations
                  and expert-level insights.
                </p>
              </div>

              <div
                className={`animate-fade-in-up ${
                  visibleElements.has("feature-2") ? "visible" : ""
                }`}
                id="feature-2"
                data-animate
                style={{
                  background: "white",
                  borderRadius: "16px",
                  padding: "2rem",
                  textAlign: "center",
                  border: "2px solid #e0f2fe",
                  transition: "all 0.3s ease",
                  boxShadow: "0 4px 15px rgba(14, 165, 233, 0.1)",
                  transform: "translateY(0px)",
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = "translateY(-5px)";
                  e.target.style.borderColor = "#0ea5e9";
                  e.target.style.boxShadow =
                    "0 8px 25px rgba(14, 165, 233, 0.15)";
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = "translateY(0px)";
                  e.target.style.borderColor = "#e0f2fe";
                  e.target.style.boxShadow =
                    "0 4px 15px rgba(14, 165, 233, 0.1)";
                }}
              >
                <div
                  style={{
                    width: "50px",
                    height: "50px",
                    background: "linear-gradient(135deg, #0ea5e9, #0284c7)",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    margin: "0 auto 1rem",
                    fontSize: "1.5rem",
                  }}
                >
                  📊
                </div>
                <h4
                  style={{
                    fontSize: "1.25rem",
                    fontWeight: "bold",
                    color: "#1e293b",
                    margin: "0 0 1rem 0",
                  }}
                >
                  Real-time Data
                </h4>
                <p
                  style={{
                    fontSize: "0.95rem",
                    color: "#64748b",
                    lineHeight: 1.6,
                    margin: "0",
                  }}
                >
                  Access live groundwater data and comprehensive resource
                  estimation across India.
                </p>
              </div>

              <div
                className={`animate-fade-in-right ${
                  visibleElements.has("feature-3") ? "visible" : ""
                }`}
                id="feature-3"
                data-animate
                style={{
                  background: "white",
                  borderRadius: "16px",
                  padding: "2rem",
                  textAlign: "center",
                  border: "2px solid #e0f2fe",
                  transition: "all 0.3s ease",
                  boxShadow: "0 4px 15px rgba(14, 165, 233, 0.1)",
                  transform: "translateY(0px)",
                }}
                onMouseOver={(e) => {
                  e.target.style.transform = "translateY(-5px)";
                  e.target.style.borderColor = "#0ea5e9";
                  e.target.style.boxShadow =
                    "0 8px 25px rgba(14, 165, 233, 0.15)";
                }}
                onMouseOut={(e) => {
                  e.target.style.transform = "translateY(0px)";
                  e.target.style.borderColor = "#e0f2fe";
                  e.target.style.boxShadow =
                    "0 4px 15px rgba(14, 165, 233, 0.1)";
                }}
              >
                <div
                  style={{
                    width: "50px",
                    height: "50px",
                    background: "linear-gradient(135deg, #0ea5e9, #0284c7)",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    margin: "0 auto 1rem",
                    fontSize: "1.5rem",
                  }}
                >
                  🏛️
                </div>
                <h4
                  style={{
                    fontSize: "1.25rem",
                    fontWeight: "bold",
                    color: "#1e293b",
                    margin: "0 0 1rem 0",
                  }}
                >
                  Official CGWB Data
                </h4>
                <p
                  style={{
                    fontSize: "0.95rem",
                    color: "#64748b",
                    lineHeight: 1.6,
                    margin: "0",
                  }}
                >
                  Powered by official Central Ground Water Board data and IIT
                  Hyderabad research.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer
        className={`animate-fade-in-up ${
          visibleElements.has("footer") ? "visible" : ""
        }`}
        id="footer"
        data-animate
        style={{
          backgroundColor: "white",
          color: "#333",
          padding: "2rem 1.5rem",
          boxShadow: "0 -2px 10px rgba(0, 0, 0, 0.1)",
          borderTop: "2px solid #e0f2fe",
        }}
      >
        <div
          style={{
            maxWidth: "1200px",
            margin: "0 auto",
            display: "flex",
            flexWrap: "wrap",
            gap: "2rem",
            alignItems: "flex-start",
          }}
        >
          {/* Left Side - Logo and Title */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
              minWidth: "300px",
              flex: "1",
            }}
          >
            {/* Logo */}
            <div
              style={{
                backgroundColor: "#1e3a8a",
                padding: "0.5rem",
                borderRadius: "50%",
                width: "50px",
                height: "50px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <img
                src="/logo1.png"
                alt="Central Ground Water Board Logo"
                style={{
                  height: "30px",
                  width: "auto",
                  objectFit: "contain",
                }}
              />
            </div>

            {/* Title */}
            <div style={{ flex: 1 }}>
              <h2
                style={{
                  fontSize: window.innerWidth > 768 ? "1.25rem" : "1rem",
                  fontWeight: "bold",
                  margin: "0 0 0.25rem 0",
                  color: "#0ea5e9",
                }}
              >
                Central Ground Water Board
              </h2>
              <p
                style={{
                  fontSize: window.innerWidth > 768 ? "0.875rem" : "0.75rem",
                  margin: "0 0 0.125rem 0",
                  color: "#666",
                }}
              >
                Department of WR, RD & GR
              </p>
              <p
                style={{
                  fontSize: window.innerWidth > 768 ? "0.75rem" : "0.65rem",
                  margin: "0",
                  color: "#888",
                }}
              >
                Ministry of Jal Shakti, Government of India
              </p>
            </div>
          </div>

          {/* Middle - Contributors */}
          <div
            style={{
              minWidth: "250px",
              flex: "1",
              textAlign: "center",
            }}
          >
            <h3
              style={{
                fontSize: "1rem",
                fontWeight: "bold",
                color: "#0ea5e9",
                margin: "0 0 1rem 0",
              }}
            >
              Development Team
            </h3>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.25rem",
              }}
            >
              <p
                style={{
                  fontSize: "0.875rem",
                  margin: "0",
                  color: "#333",
                  fontWeight: "500",
                }}
              >
                Shravya H Jain
              </p>
              <p
                style={{
                  fontSize: "0.875rem",
                  margin: "0",
                  color: "#333",
                  fontWeight: "500",
                }}
              >
                Srishyla Kumar TP
              </p>
              <p
                style={{
                  fontSize: "0.875rem",
                  margin: "0",
                  color: "#333",
                  fontWeight: "500",
                }}
              >
                Rakshita RL
              </p>
              <p
                style={{
                  fontSize: "0.875rem",
                  margin: "0",
                  color: "#333",
                  fontWeight: "500",
                }}
              >
                Shreesha S Shetty
              </p>
              <p
                style={{
                  fontSize: "0.875rem",
                  margin: "0",
                  color: "#333",
                  fontWeight: "500",
                }}
              >
                Mayuri J Shetty
              </p>
              <p
                style={{
                  fontSize: "0.875rem",
                  margin: "0",
                  color: "#333",
                  fontWeight: "500",
                }}
              >
                Mohan R
              </p>
            </div>
          </div>

          {/* Right Side - Contact Us */}
          <div
            style={{
              minWidth: "200px",
              flex: "1",
              textAlign: "right",
            }}
          >
            <h3
              style={{
                fontSize: "1rem",
                fontWeight: "bold",
                color: "#0ea5e9",
                margin: "0 0 1rem 0",
              }}
            >
              Contact Us
            </h3>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "0.5rem",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  gap: "0.5rem",
                }}
              >
                <span style={{ fontSize: "1rem" }}>📧</span>
                <a
                  href="mailto:info@cgwb.gov.in"
                  style={{
                    fontSize: "0.875rem",
                    color: "#333",
                    textDecoration: "none",
                  }}
                  onMouseOver={(e) => (e.target.style.color = "#0ea5e9")}
                  onMouseOut={(e) => (e.target.style.color = "#333")}
                >
                  info@cgwb.gov.in
                </a>
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  gap: "0.5rem",
                }}
              >
                <span style={{ fontSize: "1rem" }}>🌐</span>
                <a
                  href="https://cgwb.gov.in"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    fontSize: "0.875rem",
                    color: "#333",
                    textDecoration: "none",
                  }}
                  onMouseOver={(e) => (e.target.style.color = "#0ea5e9")}
                  onMouseOut={(e) => (e.target.style.color = "#333")}
                >
                  cgwb.gov.in
                </a>
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  gap: "0.5rem",
                }}
              >
                <span style={{ fontSize: "1rem" }}>📍</span>
                <span
                  style={{
                    fontSize: "0.875rem",
                    color: "#666",
                  }}
                >
                  New Delhi, India
                </span>
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  gap: "0.5rem",
                }}
              >
                <span style={{ fontSize: "1rem" }}>☎️</span>
                <span
                  style={{
                    fontSize: "0.875rem",
                    color: "#666",
                  }}
                >
                  +91-11-2616 5277
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Border */}
        <div
          style={{
            marginTop: "1.5rem",
            paddingTop: "1rem",
            borderTop: "1px solid #e0f2fe",
            textAlign: "center",
          }}
        >
          <p
            style={{
              fontSize: "0.75rem",
              color: "#888",
              margin: "0",
            }}
          >
            © 2024 Central Ground Water Board. All rights reserved.
          </p>
        </div>
      </footer>

      {/* Floating Chatbot Button */}
      <div
        ref={buttonRef}
        onClick={(e) => {
          console.log(
            "Chatbot clicked - isDragging:",
            isDragging,
            "hasMoved:",
            hasMoved
          );
          // Only trigger if not currently dragging and hasn't moved significantly
          if (!isDragging && !hasMoved) {
            console.log("Conditions met, calling handleChatbotClick");
            handleChatbotClick();
          } else {
            console.log("Click blocked - button was being dragged");
          }
        }}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        style={{
          position: "fixed",
          left: `${buttonPosition.x}px`,
          top: `${buttonPosition.y}px`,
          width: "60px",
          height: "60px",
          borderRadius: "50%",
          background: user
            ? "linear-gradient(135deg, #0ea5e9, #0284c7)"
            : "linear-gradient(135deg, #f59e0b, #d97706)",
          boxShadow: user
            ? "0 8px 25px rgba(14, 165, 233, 0.4)"
            : "0 8px 25px rgba(245, 158, 11, 0.4)",
          zIndex: 999,
          cursor: isDragging ? "grabbing" : "grab",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: isDragging
            ? "none"
            : "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          border: "3px solid rgba(255, 255, 255, 0.2)",
          backdropFilter: "blur(10px)",
          transform: isDragging ? "scale(1.1)" : "scale(1)",
          userSelect: "none",
        }}
        onMouseOver={(e) => {
          if (!isDragging) {
            e.target.style.transform = "translateY(-5px) scale(1.05)";
            if (user) {
              e.target.style.boxShadow = "0 15px 35px rgba(14, 165, 233, 0.6)";
              e.target.style.background =
                "linear-gradient(135deg, #0284c7, #0369a1)";
            } else {
              e.target.style.boxShadow = "0 15px 35px rgba(245, 158, 11, 0.6)";
              e.target.style.background =
                "linear-gradient(135deg, #d97706, #b45309)";
            }
          }
        }}
        onMouseOut={(e) => {
          if (!isDragging) {
            e.target.style.transform = "translateY(0) scale(1)";
            if (user) {
              e.target.style.boxShadow = "0 8px 25px rgba(14, 165, 233, 0.4)";
              e.target.style.background =
                "linear-gradient(135deg, #0ea5e9, #0284c7)";
            } else {
              e.target.style.boxShadow = "0 8px 25px rgba(245, 158, 11, 0.4)";
              e.target.style.background =
                "linear-gradient(135deg, #f59e0b, #d97706)";
            }
          }
        }}
        role="button"
        tabIndex={0}
        aria-label={
          user
            ? "Open chatbot (drag to move)"
            : "Sign in to use chatbot (drag to move)"
        }
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            handleChatbotClick();
          }
        }}
      >
        <div
          style={{
            width: "100%",
            height: "100%",
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "1.8rem",
            color: "white",
            transition: "all 0.3s ease",
            pointerEvents: "none",
            textShadow: "0 2px 4px rgba(0, 0, 0, 0.3)",
          }}
        >
          💧
        </div>
      </div>
    </div>
  );
}

export default Landing; 
