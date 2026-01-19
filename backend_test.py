#!/usr/bin/env python3
"""
Menene Backend API Test Suite
Tests all backend endpoints for the Hausa Conversational AI application
"""

import requests
import json
import uuid
from datetime import datetime
import sys
import os

# Get backend URL from frontend env
BACKEND_URL = "https://meta-tts-clone.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class MeneneAPITester:
    def __init__(self):
        self.session = requests.Session()
        self.test_user_id = str(uuid.uuid4())
        self.test_conversation_id = None
        self.test_results = []
        
    def log_test(self, test_name, success, details="", response_data=None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        if response_data and not success:
            print(f"   Response: {response_data}")
        print()
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "response": response_data
        })
    
    def test_health_check(self):
        """Test 1: Health Check Endpoint"""
        print("🔍 Testing Health Check Endpoint...")
        try:
            response = self.session.get(f"{API_BASE}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    services = data.get("services", {})
                    service_status = []
                    for service, status in services.items():
                        service_status.append(f"{service}: {status}")
                    
                    self.log_test(
                        "Health Check", 
                        True, 
                        f"Services - {', '.join(service_status)}"
                    )
                    return True
                else:
                    self.log_test("Health Check", False, "Status not healthy", data)
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_create_conversation(self):
        """Test 2: Create Conversation Endpoint"""
        print("🔍 Testing Create Conversation Endpoint...")
        try:
            payload = {
                "user_id": self.test_user_id,
                "language": "ha"
            }
            
            response = self.session.post(
                f"{API_BASE}/conversations",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("id") and data.get("user_id") == self.test_user_id:
                    self.test_conversation_id = data["id"]
                    self.log_test(
                        "Create Conversation", 
                        True, 
                        f"Created conversation ID: {self.test_conversation_id}"
                    )
                    return True
                else:
                    self.log_test("Create Conversation", False, "Invalid response structure", data)
                    return False
            else:
                self.log_test("Create Conversation", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Create Conversation", False, f"Exception: {str(e)}")
            return False
    
    def test_chat_endpoint(self):
        """Test 3: Chat Endpoint with Gemini AI"""
        print("🔍 Testing Chat Endpoint...")
        if not self.test_conversation_id:
            self.log_test("Chat Endpoint", False, "No conversation ID available")
            return False
            
        try:
            payload = {
                "conversation_id": self.test_conversation_id,
                "user_id": self.test_user_id,
                "message": "Sannu, yaya kake?",  # Hello, how are you? in Hausa
                "language": "ha"
            }
            
            response = self.session.post(
                f"{API_BASE}/chat",
                json=payload,
                timeout=30  # Longer timeout for AI response
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("response"):
                    ai_response = data["response"]
                    self.log_test(
                        "Chat Endpoint", 
                        True, 
                        f"AI Response: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}"
                    )
                    return True
                else:
                    self.log_test("Chat Endpoint", False, "Invalid response structure", data)
                    return False
            else:
                self.log_test("Chat Endpoint", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Chat Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_text_to_speech(self):
        """Test 4: Text-to-Speech Endpoint"""
        print("🔍 Testing Text-to-Speech Endpoint...")
        try:
            # Use standard English US voice since Nigerian English may not be available
            payload = {
                "text": "Hello, how are you?",  # English text
                "language": "en-US",
                "voice": "en-US-Standard-A"
            }
            
            response = self.session.post(
                f"{API_BASE}/text-to-speech",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("audio_content"):
                    audio_content = data["audio_content"]
                    cached = data.get("cached", False)
                    self.log_test(
                        "Text-to-Speech", 
                        True, 
                        f"Audio generated (cached: {cached}), length: {len(audio_content)} chars"
                    )
                    return True
                else:
                    self.log_test("Text-to-Speech", False, "Invalid response structure", data)
                    return False
            else:
                self.log_test("Text-to-Speech", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Text-to-Speech", False, f"Exception: {str(e)}")
            return False
    
    def test_get_conversations(self):
        """Test 5: Get Conversations for User"""
        print("🔍 Testing Get Conversations Endpoint...")
        try:
            response = self.session.get(
                f"{API_BASE}/conversations/{self.test_user_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "conversations" in data:
                    conversations = data["conversations"]
                    self.log_test(
                        "Get Conversations", 
                        True, 
                        f"Found {len(conversations)} conversation(s)"
                    )
                    return True
                else:
                    self.log_test("Get Conversations", False, "Invalid response structure", data)
                    return False
            else:
                self.log_test("Get Conversations", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Get Conversations", False, f"Exception: {str(e)}")
            return False
    
    def test_get_messages(self):
        """Test 6: Get Messages in Conversation"""
        print("🔍 Testing Get Messages Endpoint...")
        if not self.test_conversation_id:
            self.log_test("Get Messages", False, "No conversation ID available")
            return False
            
        try:
            response = self.session.get(
                f"{API_BASE}/conversations/{self.test_conversation_id}/messages",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "messages" in data:
                    messages = data["messages"]
                    self.log_test(
                        "Get Messages", 
                        True, 
                        f"Found {len(messages)} message(s) in conversation"
                    )
                    return True
                else:
                    self.log_test("Get Messages", False, "Invalid response structure", data)
                    return False
            else:
                self.log_test("Get Messages", False, f"HTTP {response.status_code}", response.text)
                return False
                
        except Exception as e:
            self.log_test("Get Messages", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        print("=" * 60)
        print("🚀 MENENE BACKEND API TEST SUITE")
        print("=" * 60)
        print(f"Backend URL: {BACKEND_URL}")
        print(f"Test User ID: {self.test_user_id}")
        print("=" * 60)
        print()
        
        # Run tests in order to build up test data
        tests = [
            self.test_health_check,
            self.test_create_conversation,
            self.test_chat_endpoint,
            self.test_text_to_speech,
            self.test_get_conversations,
            self.test_get_messages
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print()
        
        # Show failed tests
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print("❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['test']}: {test['details']}")
        else:
            print("✅ ALL TESTS PASSED!")
        
        print("=" * 60)
        
        return passed == total

if __name__ == "__main__":
    tester = MeneneAPITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)