import { Link, useNavigate } from 'react-router-dom'
import { auth, db, provider } from '../firebase'
import { signInWithPopup } from 'firebase/auth'
import { doc, setDoc, serverTimestamp } from 'firebase/firestore'

function Landing() {
  const navigate = useNavigate()
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

  return (
    <div className="container-centered" style={{ background: 'transparent' }}>
      <div className="glass" style={{ padding: '3rem', maxWidth: 960, width: '100%' }}>
        <div className="fade-in-up" style={{ textAlign: 'center' }}>
          <h1 className="landing-title" style={{ marginTop: 0, marginBottom: '0.5rem', color: 'var(--color-slate)' }}>Ground Water Companion</h1>
          <p className="landing-sub" style={{ marginTop: 0, color: 'var(--color-blue-gray)' }}>
            Your AI assistant for groundwater estimation insights and guidance.
          </p>

          <div style={{ marginTop: '2rem' }}>
            <button onClick={handleGoogleSignIn} style={{
              marginRight: 12,
              backgroundColor: 'var(--color-slate)',
              borderColor: 'var(--color-blue-gray)',
              color: '#0f172a'
            }}>Sign in / Sign up with Google</button>
            <Link to="/chat">
              <button style={{
                backgroundColor: 'var(--color-blue-gray)',
                borderColor: 'var(--color-slate)',
                color: '#0f172a'
              }}>
                Open Chatbot
              </button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Landing


