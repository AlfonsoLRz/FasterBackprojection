#pragma once

#include "stdafx.h"

enum class SafeQueuePushBehavior
{
    DumpIfFull,
    WaitIfFull,
    FailIfFull
};
enum class SafeQueuePopBehavior
{
    WaitIfEmpty,
    FailIfEmpty
};

// A threadsafe-queue.
template <class T, int nMaxLen>
class SafeQueue {
public:
    SafeQueue(SafeQueuePushBehavior pushBehavior = SafeQueuePushBehavior::WaitIfFull,
              SafeQueuePopBehavior popBehavior = SafeQueuePopBehavior::WaitIfEmpty)
        : _queue()
        , _mutex()
        , _condition()
        , _pushBehavior(pushBehavior)
        , _popBehavior(popBehavior)
        , _abort(false)
    {
    }

	~SafeQueue(void) = default;

    // clear the queue
    void Clear()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        while (!_queue.empty())
            _queue.pop();
    }

    // Add an element to the queue (dumping the oldest element if queue is full)
    bool Push(const T& t)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while ((int)nMaxLen > 0 && (int)_queue.size() > nMaxLen - 1) 
        {
            switch (_pushBehavior) {
            case SafeQueuePushBehavior::DumpIfFull:
                _queue.pop();
                break;
            case SafeQueuePushBehavior::WaitIfFull:
                _condition.wait(lock); // release lock as long as the wait and reaquire it afterwards.
                if (_abort) 
                {
                    _abort = false;
                    return false;
                }
                break;
            case SafeQueuePushBehavior::FailIfFull:
                return false;
            default:
                throw std::logic_error("Unknown behavior mode");
            }
        }

        _queue.push(std::move(t));
        _condition.notify_one();

        return true;
    }

    size_t Size()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

    // Get the "front"-element.
    // If the queue is empty, wait (block) until an element is avaiable.
    bool Pop(T& val)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.empty()) 
        {
            switch (_popBehavior)
            {
            case SafeQueuePopBehavior::FailIfEmpty:
                return false;
            case SafeQueuePopBehavior::WaitIfEmpty:
                // release lock as long as the wait and reaquire it afterwards.
                _condition.wait(lock);
                if (_abort) {
                    _abort = false;
                    return false;
                }
            }
        }
        val = _queue.front();
        _queue.pop();
        _condition.notify_one();
        return true;
    }

    // if a push/pop operation is blocked due to full/empty condition, this will abort the blocked wait
    void Abort()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _abort = true;
        _condition.notify_one();
    }

private:
    std::queue<T>           _queue;
    mutable std::mutex      _mutex;
    std::condition_variable _condition;
    SafeQueuePushBehavior   _pushBehavior;
    SafeQueuePopBehavior    _popBehavior;
    bool                    _abort;
};
