import { initializeApp } from 'firebase/app'
import { getAuth, GoogleAuthProvider } from 'firebase/auth'
import { getFirestore } from 'firebase/firestore'
import { getAnalytics, isSupported } from 'firebase/analytics'

const firebaseConfig = {
    apiKey: 'AIzaSyAeTuo2YxnwKhkWv1Iz2pWFx_OrSjNSDcw',
    authDomain: 'techtrek-adba1.firebaseapp.com',
    projectId: 'techtrek-adba1',
    storageBucket: 'techtrek-adba1.firebasestorage.app',
    messagingSenderId: '385921130982',
    appId: '1:385921130982:web:17cf06b7c742ed3c96f627',
    measurementId: 'G-SSQ0X8GVSR'
}

const app = initializeApp(firebaseConfig)
const auth = getAuth(app)
const db = getFirestore(app)
const provider = new GoogleAuthProvider()

isSupported().then((ok) => { if (ok) getAnalytics(app) })

export { app, auth, db, provider }