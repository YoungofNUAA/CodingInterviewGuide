package StackandQueue;

import java.util.Stack;

public class TwoStacksQueue {
	public Stack<Integer> stackPush;
	public Stack<Integer> stackPop;
	public TwoStacksQueue() {
		stackPush = new Stack<Integer>();
		stackPop = new Stack<Integer>();
	}
	public void add(int pushInt) {
		stackPush.push(pushInt);
	}
	public int poll() {
		if(stackPop.isEmpty() && stackPush.isEmpty()) {
			throw new RuntimeException("queue is empty");
		}else if(stackPop.isEmpty()) {
			while(!stackPush.isEmpty()) {
				stackPop.push(stackPush.pop());
			}
		}
		return stackPop.pop();
	}
	public int peek() {
		if(stackPop.isEmpty() && stackPush.isEmpty()) {
			throw new RuntimeException("queue is empty");
		}else if(stackPop.isEmpty()) {
			while(!stackPush.isEmpty()) {
				stackPop.push(stackPush.pop());
			}
		}
		return stackPop.peek();
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		TwoStacksQueue queue = new TwoStacksQueue();
		queue.add(1);
		queue.add(2);
		queue.add(3);
		queue.add(4);
		queue.add(5);
		System.out.println(queue.poll());
		System.out.println(queue.poll());
		//你是个二愣子
	}

}
