package StackandQueue;

import java.util.Stack;
/**
 * ͨ��һ��ջʵ����һ��ջ������
 * @author Young
 *
 */
public class SortStackByStack {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	public static void sortStackByStack(Stack<Integer>stack) {
		Stack<Integer> helper = new Stack<Integer>();
		while(!stack.isEmpty()) {
			int cur = stack.pop();
			while(!helper.isEmpty() && cur>helper.peek()) {
				stack.push(helper.pop());
			}
			helper.push(stack.pop());
		}
		while(!helper.isEmpty()) {
			stack.push(helper.pop());
		}
	}
}
