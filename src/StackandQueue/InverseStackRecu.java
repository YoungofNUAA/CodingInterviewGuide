package StackandQueue;

import java.util.Stack;

/**
 * 仅用递归函数和栈操作逆序一个栈
 * @author Young
 *
 */
public class InverseStackRecu {

	/**
	 * 
	 * @param stack
	 * @return  返回并移除栈底元素
	 */
	public int getAndRemoveLastElement(Stack<Integer> stack) {
		int result = stack.pop();
		if(stack.isEmpty()) {
			return result;
		}else {
			int last = getAndRemoveLastElement(stack);
			stack.push(result);
			return last;
		}
	}
	
	public void reverse(Stack<Integer>stack) {
		if(stack.isEmpty()) {
			return;
		}
		int i = getAndRemoveLastElement(stack);
		reverse(stack);
		stack.push(i);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//测试
//		Stack<Integer> stack = new Stack<Integer>();
//		stack.push(1);
//		stack.push(2);
//		stack.push(3);
//		InverseStackRecu test = new InverseStackRecu();
//		test.getAndRemoveLastElement(stack);
	}

}
