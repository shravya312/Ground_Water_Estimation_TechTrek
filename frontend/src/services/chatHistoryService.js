import {
    doc,
    updateDoc,
    getDoc,
    collection,
    addDoc,
    getDocs,
    query,
    orderBy,
    where,
    deleteDoc,
    serverTimestamp,
    arrayUnion,
    setDoc
} from 'firebase/firestore'
import { db } from '../firebase'

// Test Firebase connection
export const testFirebaseConnection = async() => {
    try {
        console.log('Testing Firebase connection...')
        console.log('Firebase db instance:', db)
        console.log('Firebase app:', db.app)
        return true
    } catch (error) {
        console.error('Firebase connection test failed:', error)
        return false
    }
}

// Save a chat message to user's chats subcollection
export const saveChatMessage = async(userId, message) => {
    try {
        console.log('Saving message to Firebase:', { userId, message })

        // Try subcollection approach first
        try {
            const chatsRef = collection(db, 'users', userId, 'chats')
            const docRef = await addDoc(chatsRef, {
                ...message,
                timestamp: serverTimestamp()
            })

            console.log('Message saved to subcollection with ID:', docRef.id)

            // Update user's last chat activity
            const userRef = doc(db, 'users', userId)
            await updateDoc(userRef, {
                lastChatActivity: serverTimestamp()
            })

            console.log('User activity updated successfully')
            return docRef.id
        } catch (subcollectionError) {
            console.warn('Subcollection approach failed, trying user document approach:', subcollectionError)

            // Fallback: Save to user document directly
            const userRef = doc(db, 'users', userId)
            await updateDoc(userRef, {
                chatHistory: arrayUnion({
                    ...message,
                    timestamp: serverTimestamp()
                }),
                lastChatActivity: serverTimestamp()
            })

            console.log('Message saved to user document successfully')
            return 'user_doc_' + Date.now()
        }
    } catch (error) {
        console.error('Error saving chat message:', error)
        console.error('Error details:', error.message, error.code)
        throw error
    }
}

// Get user's chat history from chats subcollection
export const getUserChatHistory = async(userId) => {
    try {
        console.log('Loading chat history for user:', userId)

        // Try subcollection approach first
        try {
            const chatsRef = collection(db, 'users', userId, 'chats')
            const q = query(chatsRef, orderBy('timestamp', 'asc'))
            const querySnapshot = await getDocs(q)

            const messages = querySnapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }))

            console.log('Loaded messages from subcollection:', messages.length, messages)
            return messages
        } catch (subcollectionError) {
            console.warn('Subcollection approach failed, trying user document approach:', subcollectionError)

            // Fallback: Get from user document
            const userRef = doc(db, 'users', userId)
            const userSnap = await getDoc(userRef)

            if (userSnap.exists()) {
                const userData = userSnap.data()
                const messages = userData.chatHistory || []
                console.log('Loaded messages from user document:', messages.length, messages)
                return messages
            }

            console.log('No chat history found')
            return []
        }
    } catch (error) {
        console.error('Error getting chat history:', error)
        console.error('Error details:', error.message, error.code)
        throw error
    }
}

// Clear user's chat history
export const clearChatHistory = async(userId) => {
    try {
        // Try subcollection approach first
        try {
            const chatsRef = collection(db, 'users', userId, 'chats')
            const querySnapshot = await getDocs(chatsRef)

            // Delete all chat documents
            const deletePromises = querySnapshot.docs.map(doc => deleteDoc(doc.ref))
            await Promise.all(deletePromises)

            console.log('Cleared subcollection chat history')
        } catch (subcollectionError) {
            console.warn('Subcollection clear failed, trying user document approach:', subcollectionError)

            // Fallback: Clear from user document
            const userRef = doc(db, 'users', userId)
            await updateDoc(userRef, {
                chatHistory: [],
                lastChatActivity: serverTimestamp()
            })

            console.log('Cleared user document chat history')
        }

        // Update user's last chat activity
        const userRef = doc(db, 'users', userId)
        await updateDoc(userRef, {
            lastChatActivity: serverTimestamp()
        })
    } catch (error) {
        console.error('Error clearing chat history:', error)
        throw error
    }
}

// Get chat history for a specific conversation (by conversation ID)
export const getConversationHistory = async(userId, conversationId) => {
    try {
        const chatsRef = collection(db, 'users', userId, 'chats')
        const q = query(
            chatsRef,
            where('conversationId', '==', conversationId),
            orderBy('timestamp', 'asc')
        )
        const querySnapshot = await getDocs(q)

        return querySnapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data()
        }))
    } catch (error) {
        console.error('Error getting conversation history:', error)
        throw error
    }
}