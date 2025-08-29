-- Initial database schema for editor-agent Supabase integration
-- This file contains the SQL commands to create the initial database structure

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable Row Level Security
ALTER DATABASE postgres SET "app.jwt_secret" TO 'your-jwt-secret-here';

-- User Profiles Table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,
    email TEXT,
    name TEXT,
    avatar_url TEXT,
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat Sessions Table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    title TEXT,
    description TEXT,
    context JSONB DEFAULT '{}',
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    message_count INTEGER DEFAULT 0,
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat Messages Table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    tool_calls JSONB DEFAULT '[]',
    tool_results JSONB DEFAULT '[]',
    parent_message_id UUID REFERENCES chat_messages(id),
    sequence_number INTEGER NOT NULL,
    tokens_used INTEGER,
    model_used TEXT,
    processing_time REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, sequence_number)
);

-- Agent Tasks Table
CREATE TABLE IF NOT EXISTS agent_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    instructions TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 5),
    context JSONB DEFAULT '{}',
    requirements TEXT[] DEFAULT '{}',
    deliverables TEXT[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    estimated_duration INTEGER, -- in minutes
    actual_duration INTEGER, -- in minutes
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    result JSONB,
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent Executions Table
CREATE TABLE IF NOT EXISTS agent_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    step_type TEXT NOT NULL,
    status TEXT DEFAULT 'started' CHECK (status IN ('started', 'running', 'completed', 'failed', 'timeout')),
    input_data JSONB,
    output_data JSONB,
    error_details JSONB,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration REAL, -- in seconds
    sequence_number INTEGER NOT NULL,
    retry_count INTEGER DEFAULT 0,
    logs TEXT[] DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(task_id, sequence_number)
);

-- File Operations Table
CREATE TABLE IF NOT EXISTS file_operations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    task_id UUID REFERENCES agent_tasks(id) ON DELETE SET NULL,
    user_id TEXT NOT NULL,
    operation_type TEXT NOT NULL CHECK (operation_type IN ('create', 'read', 'update', 'delete', 'rename', 'move')),
    file_path TEXT NOT NULL,
    old_path TEXT, -- for rename/move operations
    file_size BIGINT,
    file_hash TEXT,
    content_preview TEXT,
    metadata JSONB DEFAULT '{}',
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    backup_path TEXT,
    changes_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);
CREATE INDEX IF NOT EXISTS idx_user_profiles_is_active ON user_profiles(is_active);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_is_active ON chat_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity ON chat_sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_sequence ON chat_messages(session_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_user_id ON agent_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_session_id ON agent_tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_priority ON agent_tasks(priority);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_created_at ON agent_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_tags ON agent_tasks USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_agent_executions_task_id ON agent_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON agent_executions(status);
CREATE INDEX IF NOT EXISTS idx_agent_executions_created_at ON agent_executions(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_executions_sequence ON agent_executions(task_id, sequence_number);

CREATE INDEX IF NOT EXISTS idx_file_operations_user_id ON file_operations(user_id);
CREATE INDEX IF NOT EXISTS idx_file_operations_session_id ON file_operations(session_id);
CREATE INDEX IF NOT EXISTS idx_file_operations_task_id ON file_operations(task_id);
CREATE INDEX IF NOT EXISTS idx_file_operations_operation_type ON file_operations(operation_type);
CREATE INDEX IF NOT EXISTS idx_file_operations_file_path ON file_operations(file_path);
CREATE INDEX IF NOT EXISTS idx_file_operations_created_at ON file_operations(created_at);
CREATE INDEX IF NOT EXISTS idx_file_operations_success ON file_operations(success);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at columns
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_messages_updated_at BEFORE UPDATE ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_tasks_updated_at BEFORE UPDATE ON agent_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_executions_updated_at BEFORE UPDATE ON agent_executions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_file_operations_updated_at BEFORE UPDATE ON file_operations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create trigger to update message count in chat_sessions
CREATE OR REPLACE FUNCTION update_session_message_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE chat_sessions 
        SET message_count = message_count + 1,
            last_activity = NOW()
        WHERE id = NEW.session_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE chat_sessions 
        SET message_count = message_count - 1
        WHERE id = OLD.session_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_session_message_count_trigger
    AFTER INSERT OR DELETE ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_session_message_count();

-- Enable Row Level Security (RLS) on all tables
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE file_operations ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (basic user isolation)
-- Users can only access their own data

-- User Profiles policies
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own profile" ON user_profiles
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Chat Sessions policies
CREATE POLICY "Users can view own sessions" ON chat_sessions
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can manage own sessions" ON chat_sessions
    FOR ALL USING (auth.uid()::text = user_id);

-- Chat Messages policies
CREATE POLICY "Users can view messages in own sessions" ON chat_messages
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM chat_sessions 
            WHERE chat_sessions.id = chat_messages.session_id 
            AND chat_sessions.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage messages in own sessions" ON chat_messages
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM chat_sessions 
            WHERE chat_sessions.id = chat_messages.session_id 
            AND chat_sessions.user_id = auth.uid()::text
        )
    );

-- Agent Tasks policies
CREATE POLICY "Users can view own tasks" ON agent_tasks
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can manage own tasks" ON agent_tasks
    FOR ALL USING (auth.uid()::text = user_id);

-- Agent Executions policies
CREATE POLICY "Users can view executions of own tasks" ON agent_executions
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM agent_tasks 
            WHERE agent_tasks.id = agent_executions.task_id 
            AND agent_tasks.user_id = auth.uid()::text
        )
    );

CREATE POLICY "Users can manage executions of own tasks" ON agent_executions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM agent_tasks 
            WHERE agent_tasks.id = agent_executions.task_id 
            AND agent_tasks.user_id = auth.uid()::text
        )
    );

-- File Operations policies
CREATE POLICY "Users can view own file operations" ON file_operations
    FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can manage own file operations" ON file_operations
    FOR ALL USING (auth.uid()::text = user_id);

-- Create some useful views
CREATE OR REPLACE VIEW active_chat_sessions AS
SELECT 
    cs.*,
    up.name as user_name,
    up.email as user_email
FROM chat_sessions cs
LEFT JOIN user_profiles up ON cs.user_id = up.user_id
WHERE cs.is_active = true;

CREATE OR REPLACE VIEW recent_chat_messages AS
SELECT 
    cm.*,
    cs.title as session_title,
    cs.user_id
FROM chat_messages cm
JOIN chat_sessions cs ON cm.session_id = cs.id
WHERE cm.created_at >= NOW() - INTERVAL '7 days'
ORDER BY cm.created_at DESC;

CREATE OR REPLACE VIEW active_agent_tasks AS
SELECT 
    at.*,
    up.name as user_name,
    cs.title as session_title
FROM agent_tasks at
LEFT JOIN user_profiles up ON at.user_id = up.user_id
LEFT JOIN chat_sessions cs ON at.session_id = cs.id
WHERE at.status IN ('pending', 'in_progress');

-- Create functions for common operations
CREATE OR REPLACE FUNCTION get_user_session_count(p_user_id TEXT)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*) 
        FROM chat_sessions 
        WHERE user_id = p_user_id AND is_active = true
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION get_task_execution_summary(p_task_id UUID)
RETURNS TABLE(
    total_steps INTEGER,
    completed_steps INTEGER,
    failed_steps INTEGER,
    total_duration REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_steps,
        COUNT(CASE WHEN status = 'completed' THEN 1 END)::INTEGER as completed_steps,
        COUNT(CASE WHEN status = 'failed' THEN 1 END)::INTEGER as failed_steps,
        COALESCE(SUM(duration), 0)::REAL as total_duration
    FROM agent_executions 
    WHERE task_id = p_task_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Insert some initial data (optional)
-- This can be removed if not needed
INSERT INTO user_profiles (user_id, name, email, preferences) VALUES 
('system', 'System User', 'system@editor-agent.local', '{"theme": "dark", "notifications": true}')
ON CONFLICT (user_id) DO NOTHING;

-- Grant necessary permissions
-- Note: Adjust these based on your specific Supabase setup and requirements
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;