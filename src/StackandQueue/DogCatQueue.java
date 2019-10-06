package StackandQueue;

import java.util.LinkedList;
import java.util.Queue;

/**
 * 猫狗队列
 * @author Young
 *
 */
public class DogCatQueue {
	private Queue<PetEntryQueue> dogQ;
	private Queue<PetEntryQueue> catQ;
	private int count;
	public DogCatQueue() {
		this.dogQ = new LinkedList<PetEntryQueue>();
		this.catQ = new LinkedList<PetEntryQueue>();
		this.count = 0;
	}
	public void add(Pet pet) {
		if(pet.getType().equals("dog")) {
			dogQ.add(new PetEntryQueue(pet, this.count++));
		}else if(pet.getType().equals("cat")) {
			catQ.add(new PetEntryQueue(pet, this.count++));
		}else {
			throw new RuntimeException("没有该宠物队列");
		}
	}
	
	public void pollAll() {
		while(!dogQ.isEmpty() || !catQ.isEmpty()) {
			if(!dogQ.isEmpty() && !catQ.isEmpty()) {
				if(dogQ.peek().getCount()<catQ.peek().getCount()) {
//					return dogQ.poll().getPet();
					System.out.print(dogQ.poll());
				}else {
					System.out.print(catQ.poll());
				}
			}else if(!dogQ.isEmpty()) {
				System.out.print(dogQ.poll());
			}else if(!catQ.isEmpty()) {
				System.out.print(catQ.poll());
			}else {
				throw new RuntimeException("no cat and dogs");
			}
		}
	}
	
	public Pet pollDog() {
		if(!dogQ.isEmpty()) {
			return dogQ.poll().getPet();
		}else {
			throw new RuntimeException("dogQ is empty");
		}
	}
	
	public boolean isEmpty() {
		return dogQ.isEmpty() && catQ.isEmpty();
	}
	public boolean isDogEmpty() {
		return dogQ.isEmpty();
	}
	public boolean isCatEmpty() {
		return catQ.isEmpty();
	}
	
	public static void main(String[] args) {
		DogCatQueue dogcatqueue = new DogCatQueue();
		Pet pet1 = new Pet("dog");
		Pet pet2 = new Pet("cat");
		Pet pet3 = new Pet("dog");
		Pet pet4 = new Pet("cat");
		Pet pet5 = new Pet("dog");
		dogcatqueue.add(pet1);
		dogcatqueue.add(pet2);
		dogcatqueue.add(pet3);
		dogcatqueue.add(pet4);
		dogcatqueue.add(pet5);
		dogcatqueue.pollAll();
	}
}

class PetEntryQueue{
	private Pet pet;
	private int count;
	
	public PetEntryQueue(Pet pet,int count){
		this.pet = pet;
		this.count = count;
	}
	public String getEntryType() {
		return pet.getType();
	}
	public int getCount() {
		return this.count;
	}
	public Pet getPet() {
		return pet;
	}
	@Override
		public String toString() {
			// TODO Auto-generated method stub
			return pet.getType()+count+"-->";
		}
}

class Pet{
	private String type;
	public Pet(String type) {
		this.type = type;
	}
	public String getType() {
		return this.type;
	}
}
