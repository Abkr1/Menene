import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
  useColorScheme,
  Keyboard,
  Modal,
  ScrollView,
  Dimensions,
} from 'react-native';
import { Audio } from 'expo-av';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'http://localhost:8001';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface Conversation {
  id: string;
  user_id: string;
  title: string;
  language: string;
  created_at: string;
  updated_at: string;
}

export default function MeneneApp() {
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const insets = useSafeAreaInsets();
  
  // State
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [userId] = useState('user-' + Date.now());
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isAudioPaused, setIsAudioPaused] = useState(false);
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [sidebarVisible, setSidebarVisible] = useState(false);
  const [conversationHistory, setConversationHistory] = useState<Conversation[]>([]);
  
  const flatListRef = useRef<FlatList>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize app
  useEffect(() => {
    initializeApp();
    
    // Keyboard listeners for Android
    const keyboardDidShowListener = Keyboard.addListener(
      Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow',
      () => {
        setKeyboardVisible(true);
      }
    );
    const keyboardDidHideListener = Keyboard.addListener(
      Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide',
      () => {
        setKeyboardVisible(false);
      }
    );

    return () => {
      if (recording) {
        recording.unloadAsync();
      }
      if (sound) {
        sound.unloadAsync();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      keyboardDidShowListener.remove();
      keyboardDidHideListener.remove();
    };
  }, []);

  const initializeApp = async () => {
    try {
      // Request audio permissions
      await Audio.requestPermissionsAsync();
      
      // Set audio mode
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      // Load conversation history
      await loadConversationHistory();

      // Create or load conversation
      await createConversation();
    } catch (error) {
      console.error('Initialization error:', error);
      Alert.alert('Error', 'Failed to initialize app. Please check permissions.');
    }
  };

  const createConversation = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/conversations`, {
        user_id: userId,
        language: 'ha',
      });
      setCurrentConversation(response.data);
      // Refresh conversation history
      await loadConversationHistory();
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  // Load conversation history
  const loadConversationHistory = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/conversations/${userId}`);
      if (response.data.success) {
        setConversationHistory(response.data.conversations);
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error);
    }
  };

  // Start a new chat
  const startNewChat = async () => {
    setSidebarVisible(false);
    setMessages([]);
    setCurrentConversation(null);
    await createConversation();
  };

  // Load a specific conversation
  const loadConversation = async (conversation: Conversation) => {
    setSidebarVisible(false);
    setCurrentConversation(conversation);
    try {
      const response = await axios.get(`${BACKEND_URL}/api/conversations/${conversation.id}/messages`);
      if (response.data.success) {
        setMessages(response.data.messages);
      }
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  // Auto-name conversation based on first message
  const autoNameConversation = async (conversationId: string, firstMessage: string) => {
    try {
      // Create a short title from the first message (max 30 chars)
      const title = firstMessage.length > 30 
        ? firstMessage.substring(0, 30) + '...' 
        : firstMessage;
      
      // Update conversation title in database
      await axios.patch(`${BACKEND_URL}/api/conversations/${conversationId}`, {
        title: title
      });
      
      // Refresh history
      await loadConversationHistory();
    } catch (error) {
      console.error('Failed to auto-name conversation:', error);
    }
  };

  // Voice recording functions
  const startRecording = async () => {
    try {
      const { granted } = await Audio.getPermissionsAsync();
      if (!granted) {
        Alert.alert('Permission required', 'Microphone permission is needed to record audio.');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const { recording: newRecording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(newRecording);
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Failed to start recording:', error);
      Alert.alert('Error', 'Failed to start recording');
    }
  };

  const stopRecording = async () => {
    try {
      if (!recording) return;

      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();

      if (uri && currentConversation) {
        await transcribeAudio(uri);
      }

      setRecording(null);
      setRecordingTime(0);

    } catch (error) {
      console.error('Failed to stop recording:', error);
      Alert.alert('Error', 'Failed to stop recording');
    }
  };

  const transcribeAudio = async (audioUri: string) => {
    if (!currentConversation) return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      
      // Create file object for upload
      const audioFile: any = {
        uri: audioUri,
        type: 'audio/m4a',
        name: 'recording.m4a',
      };

      formData.append('audio', audioFile);
      formData.append('user_id', userId);
      formData.append('conversation_id', currentConversation.id);

      const response = await axios.post(
        `${BACKEND_URL}/api/speech-to-text`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      if (response.data.success) {
        const transcribedText = response.data.transcription;
        
        // Add user message to UI
        const userMessage: Message = {
          id: response.data.message_id,
          role: 'user',
          content: transcribedText,
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMessage]);

        // Get AI response
        await getAIResponse(transcribedText);
      }

    } catch (error: any) {
      console.error('Transcription error:', error);
      Alert.alert('Error', error.response?.data?.detail || 'Failed to transcribe audio');
    } finally {
      setIsLoading(false);
    }
  };

  const sendTextMessage = async () => {
    if (!inputText.trim() || !currentConversation) return;

    const isFirstMessage = messages.length === 0;
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const messageText = inputText;
    setInputText('');

    // Auto-name conversation on first message
    if (isFirstMessage) {
      await autoNameConversation(currentConversation.id, messageText);
    }

    await getAIResponse(messageText);
  };

  const getAIResponse = async (userMessage: string) => {
    if (!currentConversation) return;

    setIsLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/chat`, {
        conversation_id: currentConversation.id,
        user_id: userId,
        message: userMessage,
        language: 'ha',
      });

      if (response.data.success) {
        const assistantMessage: Message = {
          id: response.data.message_id,
          role: 'assistant',
          content: response.data.response,
          timestamp: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, assistantMessage]);

        // Generate and play TTS for assistant response
        await playTextToSpeech(response.data.response);
      }

    } catch (error: any) {
      console.error('Chat error:', error);
      Alert.alert('Error', error.response?.data?.detail || 'Failed to get response');
    } finally {
      setIsLoading(false);
    }
  };

  const playTextToSpeech = async (text: string) => {
    try {
      // Don't set isPlayingAudio here - wait until audio actually starts

      // Using TWB Voice Hausa TTS (Fully Optimized - Female Voice)
      const response = await axios.post(`${BACKEND_URL}/api/text-to-speech`, {
        text,
        language: 'ha',
      });

      if (response.data.success) {
        const audioContent = response.data.audio_content;
        
        // Create audio from base64 (WAV format from TWB Voice TTS)
        const base64Audio = `data:audio/wav;base64,${audioContent}`;
        
        // Unload previous sound
        if (sound) {
          await sound.unloadAsync();
        }

        const { sound: newSound } = await Audio.Sound.createAsync(
          { uri: base64Audio },
          { shouldPlay: true }
        );

        setSound(newSound);

        // Set playback status callback
        newSound.setOnPlaybackStatusUpdate((status: any) => {
          if (status.isPlaying && !isPlayingAudio) {
            // Audio has started playing - now show the controls
            setIsPlayingAudio(true);
          }
          if (status.didJustFinish) {
            setIsPlayingAudio(false);
            setIsAudioPaused(false);
          }
        });
      }

    } catch (error) {
      console.error('TTS error:', error);
      setIsPlayingAudio(false);
    }
  };

  // Stop audio playback
  const stopAudio = async () => {
    try {
      if (sound) {
        await sound.stopAsync();
        await sound.unloadAsync();
        setSound(null);
      }
      setIsPlayingAudio(false);
      setIsAudioPaused(false);
    } catch (error) {
      console.error('Error stopping audio:', error);
      setIsPlayingAudio(false);
      setIsAudioPaused(false);
    }
  };

  // Pause/Resume audio playback
  const togglePauseAudio = async () => {
    try {
      if (sound) {
        if (isAudioPaused) {
          await sound.playAsync();
          setIsAudioPaused(false);
        } else {
          await sound.pauseAsync();
          setIsAudioPaused(true);
        }
      }
    } catch (error) {
      console.error('Error toggling audio pause:', error);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const renderMessage = ({ item }: { item: Message }) => {
    const isUser = item.role === 'user';

    return (
      <View
        style={[
          styles.messageContainer,
          isUser ? styles.userMessage : styles.assistantMessage,
          isDark && (isUser ? styles.userMessageDark : styles.assistantMessageDark),
        ]}
      >
        <Text
          style={[
            styles.messageText,
            isUser ? styles.userMessageText : styles.assistantMessageText,
            isDark && styles.messageTextDark,
          ]}
        >
          {item.content}
        </Text>
      </View>
    );
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: isDark ? '#1a1a1a' : '#f5f5f5',
      paddingTop: insets.top,
    },
    header: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 16,
      paddingVertical: 12,
      backgroundColor: isDark ? '#1a1a1a' : '#f5f5f5',
      borderBottomWidth: 1,
      borderBottomColor: isDark ? '#333' : '#e0e0e0',
    },
    menuButton: {
      padding: 4,
    },
    headerTitle: {
      fontSize: 20,
      fontWeight: '700',
      color: '#000',
    },
    headerTitleDark: {
      color: '#fff',
    },
    headerRight: {
      width: 36,
    },
    sidebarOverlay: {
      flex: 1,
      flexDirection: 'row',
    },
    sidebarBackdrop: {
      flex: 1,
      backgroundColor: 'rgba(0,0,0,0.5)',
    },
    sidebar: {
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      width: Dimensions.get('window').width * 0.8,
      maxWidth: 320,
      backgroundColor: '#fff',
      paddingTop: insets.top + 10,
      paddingBottom: insets.bottom,
    },
    sidebarDark: {
      backgroundColor: '#1a1a1a',
    },
    sidebarHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 16,
      paddingBottom: 16,
      borderBottomWidth: 1,
      borderBottomColor: isDark ? '#333' : '#e0e0e0',
    },
    sidebarTitle: {
      fontSize: 20,
      fontWeight: '700',
      color: '#000',
    },
    sidebarTitleDark: {
      color: '#fff',
    },
    newChatButton: {
      flexDirection: 'row',
      alignItems: 'center',
      padding: 16,
      marginHorizontal: 12,
      marginTop: 12,
      backgroundColor: isDark ? '#0A84FF20' : '#007AFF10',
      borderRadius: 12,
      borderWidth: 1,
      borderColor: '#007AFF',
    },
    newChatText: {
      marginLeft: 12,
      fontSize: 16,
      fontWeight: '600',
      color: '#007AFF',
    },
    menuOptions: {
      marginTop: 16,
      paddingHorizontal: 12,
    },
    menuOption: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingVertical: 14,
      paddingHorizontal: 12,
      borderRadius: 8,
    },
    menuOptionText: {
      marginLeft: 12,
      fontSize: 16,
      color: '#333',
    },
    menuOptionTextDark: {
      color: '#ccc',
    },
    chatHistorySection: {
      flex: 1,
      marginTop: 24,
      paddingHorizontal: 12,
      borderTopWidth: 1,
      borderTopColor: isDark ? '#333' : '#e0e0e0',
      paddingTop: 16,
    },
    chatHistoryTitle: {
      fontSize: 14,
      fontWeight: '600',
      color: '#666',
      marginBottom: 12,
      paddingHorizontal: 4,
    },
    chatHistoryTitleDark: {
      color: '#888',
    },
    chatHistoryList: {
      flex: 1,
    },
    chatHistoryItem: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingVertical: 12,
      paddingHorizontal: 12,
      borderRadius: 8,
      marginBottom: 4,
    },
    chatHistoryItemDark: {
      backgroundColor: 'transparent',
    },
    chatHistoryItemActive: {
      backgroundColor: isDark ? '#0A84FF20' : '#007AFF10',
    },
    chatHistoryItemText: {
      marginLeft: 10,
      fontSize: 15,
      color: '#333',
      flex: 1,
    },
    chatHistoryItemTextDark: {
      color: '#ccc',
    },
    chatHistoryItemTextActive: {
      color: '#007AFF',
      fontWeight: '500',
    },
    noChatText: {
      fontSize: 14,
      color: '#999',
      textAlign: 'center',
      paddingVertical: 20,
    },
    noChatTextDark: {
      color: '#666',
    },
    messagesList: {
      flexGrow: 1,
      paddingHorizontal: 16,
      paddingVertical: 10,
    },
    messageContainer: {
      maxWidth: '80%',
      marginVertical: 4,
      paddingHorizontal: 16,
      paddingVertical: 10,
      borderRadius: 18,
      elevation: 1,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
    },
    userMessage: {
      alignSelf: 'flex-end',
      backgroundColor: '#007AFF',
      borderBottomRightRadius: 4,
    },
    userMessageDark: {
      backgroundColor: '#0A84FF',
    },
    assistantMessage: {
      alignSelf: 'flex-start',
      backgroundColor: '#ffffff',
      borderBottomLeftRadius: 4,
    },
    assistantMessageDark: {
      backgroundColor: '#2d2d2d',
    },
    messageText: {
      fontSize: 16,
      lineHeight: 22,
    },
    userMessageText: {
      color: '#ffffff',
    },
    assistantMessageText: {
      color: '#000000',
    },
    messageTextDark: {
      color: '#ffffff',
    },
    emptyContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      paddingHorizontal: 40,
    },
    emptyText: {
      fontSize: 18,
      color: isDark ? '#888' : '#666',
      textAlign: 'center',
      marginTop: 16,
    },
    emptySubtext: {
      fontSize: 14,
      color: isDark ? '#666' : '#999',
      textAlign: 'center',
      marginTop: 8,
    },
    centeredInputWrapper: {
      position: 'absolute',
      bottom: 0,
      left: 0,
      right: 0,
      top: 0,
      justifyContent: 'center',
      alignItems: 'center',
      pointerEvents: 'box-none',
    },
    welcomeSection: {
      alignItems: 'center',
      marginBottom: 32,
    },
    inputContainer: {
      width: '90%',
      maxWidth: 600,
      paddingHorizontal: 16,
      paddingVertical: 12,
      backgroundColor: isDark ? '#2d2d2d' : '#ffffff',
      borderRadius: 24,
      elevation: 4,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.2,
      shadowRadius: 8,
    },
    bottomInputContainer: {
      paddingHorizontal: 16,
      paddingVertical: 12,
      paddingBottom: keyboardVisible ? 12 : Math.max(insets.bottom, 12),
      backgroundColor: isDark ? '#2d2d2d' : '#ffffff',
      borderTopWidth: 1,
      borderTopColor: isDark ? '#404040' : '#e0e0e0',
    },
    recordingIndicator: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      paddingVertical: 12,
      backgroundColor: isDark ? '#1a1a1a' : '#f9f9f9',
      borderRadius: 12,
      marginBottom: 8,
    },
    recordingDot: {
      width: 12,
      height: 12,
      borderRadius: 6,
      backgroundColor: '#FF3B30',
      marginRight: 8,
    },
    recordingText: {
      fontSize: 16,
      fontWeight: '600',
      color: isDark ? '#ffffff' : '#000000',
      marginRight: 8,
    },
    inputRow: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
    },
    textInput: {
      flex: 1,
      minHeight: 40,
      maxHeight: 100,
      backgroundColor: isDark ? '#1a1a1a' : '#f5f5f5',
      borderRadius: 20,
      paddingHorizontal: 16,
      paddingVertical: 10,
      fontSize: 16,
      color: isDark ? '#ffffff' : '#000000',
    },
    iconButton: {
      width: 44,
      height: 44,
      borderRadius: 22,
      backgroundColor: '#007AFF',
      alignItems: 'center',
      justifyContent: 'center',
    },
    iconButtonRecording: {
      backgroundColor: '#FF3B30',
    },
    iconButtonDisabled: {
      backgroundColor: '#ccc',
    },
    loadingContainer: {
      paddingVertical: 8,
      alignItems: 'center',
    },
    audioPlayingContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      paddingHorizontal: 16,
      paddingVertical: 10,
      backgroundColor: isDark ? '#1a3a5c' : '#e3f2fd',
      borderTopWidth: 1,
      borderTopColor: isDark ? '#2a4a6c' : '#90caf9',
    },
    audioPlayingIndicator: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    audioPlayingText: {
      marginLeft: 8,
      fontSize: 14,
      color: '#007AFF',
      fontWeight: '500',
    },
    audioPlayingTextDark: {
      color: '#64b5f6',
    },
    audioControlButtons: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
    },
    pauseAudioButton: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 12,
      paddingVertical: 6,
      backgroundColor: isDark ? '#1a3a5c' : '#e3f2fd',
      borderRadius: 16,
      borderWidth: 1,
      borderColor: '#007AFF',
    },
    pauseAudioText: {
      marginLeft: 4,
      fontSize: 14,
      color: '#007AFF',
      fontWeight: '600',
    },
    stopAudioButton: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingHorizontal: 12,
      paddingVertical: 6,
      backgroundColor: isDark ? '#4a1a1a' : '#ffebee',
      borderRadius: 16,
      borderWidth: 1,
      borderColor: '#FF3B30',
    },
    stopAudioText: {
      marginLeft: 4,
      fontSize: 14,
      color: '#FF3B30',
      fontWeight: '600',
    },
    emptyContainer: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      paddingHorizontal: 40,
    },
    emptyText: {
      fontSize: 18,
      color: isDark ? '#888' : '#666',
      textAlign: 'center',
      marginTop: 16,
    },
    emptySubtext: {
      fontSize: 14,
      color: isDark ? '#666' : '#999',
      textAlign: 'center',
      marginTop: 8,
    },
  });

  return (
    <View style={styles.container}>
      {/* Header with Menu Button */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.menuButton}
          onPress={() => setSidebarVisible(true)}
        >
          <Ionicons name="menu" size={28} color={isDark ? '#fff' : '#000'} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, isDark && styles.headerTitleDark]}>Menene</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Sidebar Modal */}
      <Modal
        visible={sidebarVisible}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setSidebarVisible(false)}
      >
        <View style={styles.sidebarOverlay}>
          <TouchableOpacity 
            style={styles.sidebarBackdrop} 
            activeOpacity={1}
            onPress={() => setSidebarVisible(false)}
          />
          <View style={[styles.sidebar, isDark && styles.sidebarDark]}>
            {/* Sidebar Header */}
            <View style={styles.sidebarHeader}>
              <Text style={[styles.sidebarTitle, isDark && styles.sidebarTitleDark]}>Menu</Text>
              <TouchableOpacity onPress={() => setSidebarVisible(false)}>
                <Ionicons name="close" size={28} color={isDark ? '#fff' : '#000'} />
              </TouchableOpacity>
            </View>

            {/* New Chat Button */}
            <TouchableOpacity style={styles.newChatButton} onPress={startNewChat}>
              <Ionicons name="add-circle-outline" size={24} color="#007AFF" />
              <Text style={styles.newChatText}>New Chat</Text>
            </TouchableOpacity>

            {/* Menu Options */}
            <View style={styles.menuOptions}>
              <TouchableOpacity style={styles.menuOption}>
                <Ionicons name="person-outline" size={22} color={isDark ? '#aaa' : '#666'} />
                <Text style={[styles.menuOptionText, isDark && styles.menuOptionTextDark]}>Account</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.menuOption}>
                <Ionicons name="information-circle-outline" size={22} color={isDark ? '#aaa' : '#666'} />
                <Text style={[styles.menuOptionText, isDark && styles.menuOptionTextDark]}>About</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.menuOption}>
                <Ionicons name="settings-outline" size={22} color={isDark ? '#aaa' : '#666'} />
                <Text style={[styles.menuOptionText, isDark && styles.menuOptionTextDark]}>Settings</Text>
              </TouchableOpacity>
            </View>

            {/* Chat History */}
            <View style={styles.chatHistorySection}>
              <Text style={[styles.chatHistoryTitle, isDark && styles.chatHistoryTitleDark]}>
                Chat History
              </Text>
              <ScrollView style={styles.chatHistoryList} showsVerticalScrollIndicator={false}>
                {conversationHistory.length === 0 ? (
                  <Text style={[styles.noChatText, isDark && styles.noChatTextDark]}>
                    No previous chats
                  </Text>
                ) : (
                  conversationHistory.map((conv) => (
                    <TouchableOpacity
                      key={conv.id}
                      style={[
                        styles.chatHistoryItem,
                        currentConversation?.id === conv.id && styles.chatHistoryItemActive,
                        isDark && styles.chatHistoryItemDark,
                      ]}
                      onPress={() => loadConversation(conv)}
                    >
                      <Ionicons 
                        name="chatbubble-outline" 
                        size={18} 
                        color={currentConversation?.id === conv.id ? '#007AFF' : (isDark ? '#aaa' : '#666')} 
                      />
                      <Text 
                        style={[
                          styles.chatHistoryItemText,
                          currentConversation?.id === conv.id && styles.chatHistoryItemTextActive,
                          isDark && styles.chatHistoryItemTextDark,
                        ]}
                        numberOfLines={1}
                      >
                        {conv.title || 'New Conversation'}
                      </Text>
                    </TouchableOpacity>
                  ))
                )}
              </ScrollView>
            </View>
          </View>
        </View>
      </Modal>

      {messages.length === 0 ? (
        /* Empty state - centered welcome and input */
        <View style={styles.centeredInputWrapper}>
          <View style={styles.welcomeSection}>
            <Ionicons
              name="chatbubbles-outline"
              size={80}
              color={isDark ? '#404040' : '#ccc'}
            />
            <Text style={styles.emptyText}>Barka da zuwa! Menene a ke bukata?</Text>
            <Text style={styles.emptySubtext}>
              Start a conversation by typing or using your voice
            </Text>
          </View>

          <View style={styles.inputContainer}>
            {isRecording && (
              <View style={styles.recordingIndicator}>
                <View style={styles.recordingDot} />
                <Text style={styles.recordingText}>Recording</Text>
                <Text style={styles.recordingText}>{formatTime(recordingTime)}</Text>
              </View>
            )}

            <View style={styles.inputRow}>
              <TextInput
                style={styles.textInput}
                placeholder="Type in Hausa..."
                placeholderTextColor={isDark ? '#666' : '#999'}
                value={inputText}
                onChangeText={setInputText}
                multiline
                editable={!isLoading && !isRecording}
              />

              <TouchableOpacity
                style={[
                  styles.iconButton,
                  isRecording && styles.iconButtonRecording,
                  isLoading && styles.iconButtonDisabled,
                ]}
                onPress={isRecording ? stopRecording : startRecording}
                disabled={isLoading}
              >
                <Ionicons
                  name={isRecording ? 'stop' : 'mic'}
                  size={24}
                  color="#ffffff"
                />
              </TouchableOpacity>

              {inputText.trim().length > 0 && (
                <TouchableOpacity
                  style={[
                    styles.iconButton,
                    isLoading && styles.iconButtonDisabled,
                  ]}
                  onPress={sendTextMessage}
                  disabled={isLoading || isRecording}
                >
                  <Ionicons name="send" size={20} color="#ffffff" />
                </TouchableOpacity>
              )}
            </View>
          </View>
        </View>
      ) : (
        /* Messages exist - normal layout with input at bottom */
        <KeyboardAvoidingView 
          style={{ flex: 1 }}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
        >
          <FlatList
            ref={flatListRef}
            data={messages}
            renderItem={renderMessage}
            keyExtractor={(item) => item.id}
            style={{ flex: 1 }}
            contentContainerStyle={styles.messagesList}
            onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
            onLayout={() => flatListRef.current?.scrollToEnd({ animated: false })}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={true}
          />

          {/* Loading Indicator */}
          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="small" color="#007AFF" />
            </View>
          )}

          {/* Audio Playing Indicator with Pause/Play and Stop Buttons */}
          {isPlayingAudio && (
            <View style={styles.audioPlayingContainer}>
              <View style={styles.audioPlayingIndicator}>
                <Ionicons name={isAudioPaused ? "volume-mute" : "volume-high"} size={20} color="#007AFF" />
                <Text style={[styles.audioPlayingText, isDark && styles.audioPlayingTextDark]}>
                  {isAudioPaused ? 'Paused' : 'Playing audio...'}
                </Text>
              </View>
              <View style={styles.audioControlButtons}>
                <TouchableOpacity
                  style={styles.pauseAudioButton}
                  onPress={togglePauseAudio}
                >
                  <Ionicons name={isAudioPaused ? "play-circle" : "pause-circle"} size={28} color="#007AFF" />
                  <Text style={styles.pauseAudioText}>{isAudioPaused ? 'Play' : 'Pause'}</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.stopAudioButton}
                  onPress={stopAudio}
                >
                  <Ionicons name="stop-circle" size={28} color="#FF3B30" />
                  <Text style={styles.stopAudioText}>Stop</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* Input at bottom */}
          <View style={styles.bottomInputContainer}>
            {isRecording && (
              <View style={styles.recordingIndicator}>
                <View style={styles.recordingDot} />
                <Text style={styles.recordingText}>Recording</Text>
                <Text style={styles.recordingText}>{formatTime(recordingTime)}</Text>
              </View>
            )}

            <View style={styles.inputRow}>
              <TextInput
                style={styles.textInput}
                placeholder="Type in Hausa..."
                placeholderTextColor={isDark ? '#666' : '#999'}
                value={inputText}
                onChangeText={setInputText}
                multiline
                editable={!isLoading && !isRecording}
              />

              <TouchableOpacity
                style={[
                  styles.iconButton,
                  isRecording && styles.iconButtonRecording,
                  isLoading && styles.iconButtonDisabled,
                ]}
                onPress={isRecording ? stopRecording : startRecording}
                disabled={isLoading}
              >
                <Ionicons
                  name={isRecording ? 'stop' : 'mic'}
                  size={24}
                  color="#ffffff"
                />
              </TouchableOpacity>

              {inputText.trim().length > 0 && (
                <TouchableOpacity
                  style={[
                    styles.iconButton,
                    isLoading && styles.iconButtonDisabled,
                  ]}
                  onPress={sendTextMessage}
                  disabled={isLoading || isRecording}
                >
                  <Ionicons name="send" size={20} color="#ffffff" />
                </TouchableOpacity>
              )}
            </View>
          </View>
        </KeyboardAvoidingView>
      )}
    </View>
  );
}
