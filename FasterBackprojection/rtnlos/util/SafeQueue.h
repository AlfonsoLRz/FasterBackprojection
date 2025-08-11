#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace NLOS {

    enum class SafeQueuePushBehavior {
        DumpIfFull,
        WaitIfFull,
        FailIfFull
    };
    enum class SafeQueuePopBehavior {
        WaitIfEmpty,
        FailIfEmpty
    };

    // A threadsafe-queue.
    template <class T, int nMaxLen>
    class SafeQueue {
    public:
        SafeQueue(SafeQueuePushBehavior pushBehavior = SafeQueuePushBehavior::WaitIfFull,
                  SafeQueuePopBehavior popBehavior = SafeQueuePopBehavior::WaitIfEmpty)
            : m_q()
            , m_m()
            , m_c()
            , m_pushBehavior(pushBehavior)
            , m_popBehavior(popBehavior)
            , m_abort(false)
        {
        }

        ~SafeQueue(void)
        {
        }

        // clear the queue
        void Clear()
        {
            std::lock_guard<std::mutex> lock(m_m);
            while (!m_q.empty())
                m_q.pop();
        }

        // Add an element to the queue (dumping the oldest element if queue is full)
        bool Push(const T& t)
        {
            std::unique_lock<std::mutex> lock(m_m);
            while ((int)nMaxLen > 0 && (int)m_q.size() > nMaxLen - 1) {
                switch (m_pushBehavior) {
                case SafeQueuePushBehavior::DumpIfFull:
                    m_q.pop();
                    break;
                case SafeQueuePushBehavior::WaitIfFull:
                    m_c.wait(lock); // release lock as long as the wait and reaquire it afterwards.
                    if (m_abort) {
                        m_abort = false;
                        return false;
                    }
                    break;
                case SafeQueuePushBehavior::FailIfFull:
                    return false;
                default:
                    throw std::logic_error("Unknown behavior mode");
                }
            }
            m_q.push(std::move(t));
            m_c.notify_one();
            return true;
        }

         size_t Size() {
             std::lock_guard<std::mutex> lock(m_m);
             return m_q.size();
         }

        // Get the "front"-element.
        // If the queue is empty, wait (block) until an element is avaiable.
        bool Pop(T& val)
        {
            std::unique_lock<std::mutex> lock(m_m);
            while (m_q.empty()) {
                switch (m_popBehavior) {
                case SafeQueuePopBehavior::FailIfEmpty:
                    return false;
                case SafeQueuePopBehavior::WaitIfEmpty:
                    // release lock as long as the wait and reaquire it afterwards.
                    m_c.wait(lock);
                    if (m_abort) {
                        m_abort = false;
                        return false;
                    }
                }
            }
            val = m_q.front();
            m_q.pop();
            m_c.notify_one();
            return true;
        }

        // if a push/pop operation is blocked due to full/empty condition, this will abort the blocked wait
        void Abort() {
            std::lock_guard<std::mutex> lock(m_m);
            m_abort = true;
            m_c.notify_one();
        }

    private:
        std::queue<T> m_q;
        mutable std::mutex m_m;
        std::condition_variable m_c;
        SafeQueuePushBehavior m_pushBehavior;
        SafeQueuePopBehavior m_popBehavior;
        bool m_abort;
    };
}
