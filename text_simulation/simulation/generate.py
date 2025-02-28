from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Any
import random
from typing import TypeVar, cast
from inspect import isabstract
import re
import os
import numpy as np

DIR = os.path.dirname(__file__)
MAX_ITER = 100

class Predicate:
	def __init__(self, name: str, parameter_list: list[str]) -> None:
		self.name = name
		self.parameter_list = parameter_list
	
	def __str__(self) -> str:
		return "({} {})".format(self.name, " ".join(self.parameter_list))

class Action:
	def __init__(self, name: str, parameter_list: list[str], preconditions: list[str], effects: list[str]) -> None:
		self.name = name
		self.parameter_list = parameter_list
		self.preconditions = preconditions
		self.effects = effects
	
	def __str__(self) -> str:
		preconditions = "\t\t:precondition (and\n" \
							+ "\t\t\t({})\n".format(")\n\t\t\t(".join(self.preconditions)) \
						+ "\t\t)\n" if len(self.preconditions) > 0 else ""
		return f"\t(:action {self.name}\n" \
					+ "\t\t:parameters ({})\n".format(" ".join(self.parameter_list)) \
					+ preconditions \
					+ "\t\t:effect (and\n" \
					+ "\t\t\t({})\n".format(")\n\t\t\t(".join(self.effects)) \
					+ "\t\t)\n" \
					+ "\t)\n"
	
class Goal:
	def __init__(self, description: str, predicate_list: list[str]) -> None:
		self.description = description
		self.predicate_list = predicate_list
	
	def __str__(self) -> str:
		return f"\t(:goal\n" \
					+ "\t\t(and\n" \
						+ "\t\t\t({})\n".format(")\n\t\t\t(".join(self.predicate_list)) \
					+ "\t\t)\n" \
				+ "\t)\n"

class EntityID:
	def __init__(self, name: str, concept: str):
		self.name = name
		self.concept = concept
	
	def __str__(self) -> str:
		return f'instance: ["{self.name}", "{self.concept}"]'

class Attribute:
	def __init__(self, name: str, value: EntityID | int | str | bool | float) -> None:
		self.name = name
		self.value = value
	
	def to_yaml(self, num_indent: int) -> str:
		indent = "  " * num_indent
		if isinstance(self.value, str):
			str_value = f'"{self.value}"'
		elif isinstance(self.value, EntityID):
			str_value = f"\n{indent}    {self.value}"
		else:
			str_value = str(self.value).lower()
		return f"{indent}- name: {self.name}\n" \
			   f"{indent}  value: {str_value}"

class Instance:
	def __init__(self, entity_id: EntityID, attributes: list[Attribute]):
		self.entity_id = entity_id
		self.attributes = attributes

	def to_yaml(self, num_indent: int) -> str:
		indent = "  " * num_indent
		yaml = f"{indent}- {self.entity_id}\n"
		if len(self.attributes) > 0:
			yaml += f"{indent}  attributes:\n"
			for attribute in self.attributes:
				yaml += attribute.to_yaml(num_indent + 1) + "\n"
		return yaml

class RoomItem(ABC):
	def initialize_entity_id(self):
		self.entity_id = EntityID(self.token_name, self.get_type_name())

	def __init__(self, name: str, token_name: str) -> None:
		self.name = name
		self.token_name = token_name
		self.initialize_entity_id()

	@abstractmethod
	def perform_action(self, people: list[Person]) -> str | None:
		pass

	@staticmethod
	@abstractmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		pass

	@staticmethod
	@abstractmethod
	def get_pddl_domain_actions() -> list[Action]:
		pass

	@classmethod
	def get_type_name(cls) -> str:
		return cls.__name__.lower()

	@classmethod
	def get_required_types(cls) -> list[str]:
		return [cls.get_type_name() + " - object"]
	
	@abstractmethod
	def get_init_conditions(self) -> list[str]:
		pass

	def get_pddl_objects(self) -> list[str]:
		return [self.token_name + " - " + self.get_type_name()]
	
	@staticmethod
	def get_static_entities() -> list[Instance]:
		return []
	
	@abstractmethod
	def get_yaml_attributes(self) -> list[Attribute]:
		pass
	
	def get_yaml_instance(self) -> Instance:
		return Instance(self.entity_id, self.get_yaml_attributes())
	
	@abstractmethod
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		pass

class Queryable:
	@abstractmethod
	def generate_query_answer(self) -> tuple[str, str]:
		pass

class StationaryItem(RoomItem):
	def __init__(self, name: str, parent: Room) -> None:
		suffix = re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()
		token_name = parent.token_name + "_" + suffix
		super().__init__(name, token_name)
		self.parent = parent
	
	@staticmethod
	@abstractmethod
	def generate_instance(parent: Room) -> tuple[StationaryItem, list[AccompanyingItem]]:
		pass

	@abstractmethod
	def get_description(self) -> str:
		pass

	def get_full_name_with_room(self) -> str:
		return f"{self.name} in {self.parent.name}"
	
	def get_init_conditions(self) -> list[str]:
		return [Room.get_in_room_predicate(self.parent.token_name, self.token_name)]
	
	def get_yaml_attributes(self) -> list[Attribute]:
		return []

class MovableItem(RoomItem, Queryable):
	def __init__(self, name: str, token_name: str, shortened_name: str, use_default_article: bool = True) -> None:
		super().__init__(name, token_name)
		self.set_shortened_name(shortened_name, use_default_article)
		self.container: Container | Person
		self.relative_location: str | None = None
		self.extra_location_info: dict[Any, Any] = {}
	
	def generate_query_answer(self) -> tuple[str, str]:
		query = f"Where is {self.shortened_name}?"
		if isinstance(self.container, Person):
			answer = f"{self.container.name} is holding {self.shortened_name}."
		else:
			answer = f"{self.shortened_name.capitalize()} is {self.relative_location} the {self.container.get_full_name_with_room()}."
		return query, answer
	
	def perform_action(self, people: list[Person]) -> str | None:
		return None

	@staticmethod
	@abstractmethod
	def generate_instance() -> MovableItem | None:
		pass

	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return []
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		return []
	
	def get_init_conditions(self) -> list[str]:
		if isinstance(self.container, Person):
			return [Person.get_in_hand_predicate(self.token_name, self.container.token_name)]
		return self.container.get_contains_predicates(self.container.token_name, self.token_name, **self.extra_location_info)
	
	def get_yaml_attributes(self) -> list[Attribute]:
		attributes = [Attribute(Person.IN_HAND_RELATION if isinstance(self.container, Person) else self.container.get_contains_relation(), self.container.entity_id)]
		if "extra_attributes" in self.extra_location_info.keys():
			extras = self.extra_location_info.get("extra_attributes")
			assert isinstance(extras, list)
			attributes += extras
		return attributes
	
	def set_shortened_name(self, shortened_name: str, use_default_article: bool) -> None:
		self.shortened_name = "{}{}".format("the " if use_default_article else "", shortened_name)
	
	def exchange_container(self, new_container: Container | Person) -> None:
		self.container.items.remove(self)
		new_container.items.append(self)
		self.container = new_container
		if isinstance(new_container, Person):
			self.relative_location = None
			self.extra_location_info = {}
		else:
			self.relative_location, self.extra_location_info = new_container.generate_relative_location()
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		usable_people = people.copy()
		person = usable_people.pop(random.randrange(len(usable_people)))
		while self in person.items:
			if len(usable_people) == 0:
				return None
			person = usable_people.pop(random.randrange(len(usable_people)))
		agent.parent = person.parent
		self.exchange_container(person)
		return Goal(
			f"Hand {person.name} {self.shortened_name}.",
			[person.get_in_hand_predicate(self.token_name, person.token_name)]
		)

class AccompanyingItem(MovableItem):
	def __init__(self, name: str, token_name: str, shortened_name: str, use_default_article: bool = True) -> None:
		super().__init__(name, token_name, shortened_name, use_default_article)
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		return None

class Container(StationaryItem):
	ITEM_PARAM = "?a"
	CONTAINER_PARAM = "?b"
	AGENT_PARAM = "?c"
	ROOM_PARAM = "?d"
	EXTRA_INFO: dict[str, Any] = {}

	def __init__(self, name: str, parent: Room) -> None:
		super().__init__(name, parent)
		self.items: list[MovableItem] = []

	@staticmethod
	@abstractmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		pass

	@classmethod
	def get_holdable_types(cls) -> list[type[MovableItem]]:
		return [movable_type for movable_type in movable_types if cls.can_hold(movable_type)]
	
	def populate(self, items: list[MovableItem], max_allowed: int) -> None:
		holdables = [item for item in items if self.can_hold(type(item))]
		random.shuffle(holdables)
		selected = 0
		while len(holdables) > 0 and selected < max_allowed:
			item = holdables.pop()
			items.remove(item)
			self.items.append(item)
			selected += 1
		for item in self.items:
			item.container = self
			item.relative_location, item.extra_location_info = self.generate_relative_location()
	
	def get_description(self) -> str:
		if len(self.items) == 0:
			return f"The {self.name} is empty. "
		return f"The {self.name} has {self.get_item_list_description(self.items)}. "
	
	@staticmethod
	def get_item_list_description(item_list: list[MovableItem]) -> str:
		description = ""
		for i, item in enumerate(item_list):
			description += "a{} {}".format("n" if item.name[0] in "aeiou" else "", item.name)
			if len(item_list) == 2 and i == 0:
				description += " and "
			else:
				if i < len(item_list) - 1 and (i != len(item_list) - 2 or len(item_list) > 2):
					description += ", "
				if i == len(item_list) - 2:
					description += "and "
		return description
	
	@abstractmethod
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		pass

	def perform_action(self, people: list[Person]) -> str | None:
		person = random.choice(people)
		items = person.items.copy()
		random.shuffle(items)
		for item in items:
			if not self.can_hold(type(item)):
				continue
			person.parent = self.parent
			item.exchange_container(self)
			return f"{person.name} went to {self.parent.name} and placed {item.shortened_name} {item.relative_location} the {self.name}."
		return None
		
	@classmethod
	def get_contains_relation(cls) -> str:
		return f"placed_at_{cls.get_type_name()}"

	@classmethod
	def get_place_action_name(cls) -> str:
		return f"place_at_{cls.get_type_name()}"

	@classmethod
	def get_remove_action_name(cls) -> str:
		return f"remove_from_{cls.get_type_name()}"
	
	@classmethod
	def get_contains_predicates(cls, container_param: str, item_param: str, **kwargs) -> list[str]:
		return [f"{cls.get_contains_relation()} {item_param} {container_param}"]
	
	@classmethod
	def get_holdable_param(cls, param_token: str) -> str:
		holdable_types = [holdable_type.get_type_name() for holdable_type in cls.get_holdable_types()]
		return "{} - (either {})".format(param_token, " ".join(holdable_types))

	@classmethod
	def get_default_param_list(cls) -> list[str]:
		return [cls.get_holdable_param(cls.ITEM_PARAM), f"{cls.CONTAINER_PARAM} - {cls.get_type_name()}", f"{cls.AGENT_PARAM} - {Agent.TYPE_NAME}", f"{cls.ROOM_PARAM} - {Room.TYPE_NAME}"]
	
	@classmethod
	def get_pddl_domain_predicates(cls) -> list[Predicate]:
		return [Predicate(cls.get_contains_relation(), [cls.get_holdable_param(cls.ITEM_PARAM), f"{cls.CONTAINER_PARAM} - {cls.get_type_name()}"])]
	
	@classmethod
	def get_place_action(cls) -> Action:
		param_list = cls.get_default_param_list()
		holding_predicate = Agent.get_in_hand_predicate(cls.ITEM_PARAM, cls.AGENT_PARAM)
		contains_predicates = cls.get_contains_predicates(cls.CONTAINER_PARAM, cls.ITEM_PARAM, **cls.EXTRA_INFO)

		place_preconditions = [
			holding_predicate,
			Room.get_in_room_predicate(cls.ROOM_PARAM, cls.CONTAINER_PARAM),
			Agent.get_in_room_predicate(cls.AGENT_PARAM, cls.ROOM_PARAM),
		]
		place_effects = [f"not ({holding_predicate})"] + contains_predicates

		return Action(cls.get_place_action_name(), param_list, place_preconditions, place_effects)
	
	@classmethod
	def get_remove_action(cls) -> Action:
		param_list = cls.get_default_param_list()
		holding_predicate = Agent.get_in_hand_predicate(cls.ITEM_PARAM, cls.AGENT_PARAM)
		contains_predicates = cls.get_contains_predicates(cls.CONTAINER_PARAM, cls.ITEM_PARAM, **cls.EXTRA_INFO)

		remove_preconditions = contains_predicates + [
			Room.get_in_room_predicate(cls.ROOM_PARAM, cls.CONTAINER_PARAM),
			Agent.get_in_room_predicate(cls.AGENT_PARAM, cls.ROOM_PARAM),
		]
		remove_effects = [f"not ({pred})" for pred in contains_predicates] + [holding_predicate]

		return Action(cls.get_remove_action_name(), param_list, remove_preconditions, remove_effects)
	
	@classmethod
	def get_pddl_domain_actions(cls) -> list[Action]:
		return [cls.get_place_action(), cls.get_remove_action()]
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		random.shuffle(all_items)
		for item in all_items:
			if not self.can_hold(type(item)):
				continue
			agent.parent = self.parent
			item.exchange_container(self)
			return Goal(
				f"Place {item.shortened_name} {item.relative_location} the {self.get_full_name_with_room()}.",
				self.get_contains_predicates(self.token_name, item.token_name, **item.extra_location_info)
			)
		return None

class InteractableItem(RoomItem, Queryable):
	@abstractmethod
	def get_special_init_conditions(self) -> list[str]:
		pass

	@abstractmethod
	def get_special_yaml_attributes(self) -> list[Attribute]:
		pass

class StationaryInteractable(StationaryItem, InteractableItem):
	def get_init_conditions(self) -> list[str]:
		return StationaryItem.get_init_conditions(self) + self.get_special_init_conditions()
	
	def get_yaml_attributes(self) -> list[Attribute]:
		return StationaryItem.get_yaml_attributes(self) + self.get_special_yaml_attributes()

class MovableInteractable(MovableItem, InteractableItem):
	@abstractmethod
	def generate_interactable_qa(self) -> tuple[str, str]:
		pass

	def generate_query_answer(self) -> tuple[str, str]:
		return self.generate_interactable_qa() if random.choice([True, False]) else MovableItem.generate_query_answer(self)
	
	@abstractmethod
	def interact(self, people: list[Person]) -> str | None:
		pass

	@staticmethod
	@abstractmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		pass

	@staticmethod
	@abstractmethod
	def get_pddl_domain_actions() -> list[Action]:
		pass

	def perform_action(self, people: list[Person]) -> str | None:
		for _ in range(MAX_ITER):
			action = self.interact(people) if random.choice([True, False]) else MovableItem.perform_action(self, people)
			if action is not None:
				return action
		raise Exception("Unable to generate action")

	def get_init_conditions(self) -> list[str]:
		return MovableItem.get_init_conditions(self) + self.get_special_init_conditions()

	def get_yaml_attributes(self) -> list[Attribute]:
		return MovableItem.get_yaml_attributes(self) + self.get_special_yaml_attributes()

class InteractableContainer(Container, StationaryInteractable):
	@abstractmethod
	def interact(self, people: list[Person]) -> str | None:
		pass

	def perform_action(self, people: list[Person]) -> str | None:
		for _ in range(MAX_ITER):
			action = self.interact(people) if random.choice([True, False]) else Container.perform_action(self, people)
			if action is not None:
				return action
		raise Exception("Unable to generate action")
	
	@abstractmethod
	def generate_interactable_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		pass

	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		if random.choice([True, False]):
			goal = self.generate_interactable_goal(people, all_items, agent)
			if goal is None:
				goal = Container.generate_goal(self, people, all_items, agent)
			return goal
		goal = Container.generate_goal(self, people, all_items, agent)
		if goal is None:
			goal = self.generate_interactable_goal(people, all_items, agent)
		return goal
	
	def get_init_conditions(self) -> list[str]:
		return Container.get_init_conditions(self) + self.get_special_init_conditions()
	
	def get_yaml_attributes(self) -> list[Attribute]:
		return Container.get_yaml_attributes(self) + self.get_special_yaml_attributes()
	
	@abstractmethod
	def get_interactable_description(self) -> str:
		pass

	def get_container_description(self) -> str:
		return Container.get_description(self)
	
	def get_description(self) -> str:
		return self.get_container_description() + self.get_interactable_description()

	@staticmethod
	@abstractmethod
	def get_special_domain_predicates() -> list[Predicate]:
		pass

	@classmethod
	def get_pddl_domain_predicates(cls) -> list[Predicate]:
		return super().get_pddl_domain_predicates() + cls.get_special_domain_predicates()
	
	@staticmethod
	@abstractmethod
	def get_special_domain_actions() -> list[Action]:
		pass

	@classmethod
	def get_pddl_domain_actions(cls) -> list[Action]:
		return super().get_pddl_domain_actions() + cls.get_special_domain_actions()

class Table(Container):
	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return True
	
	@staticmethod
	def generate_instance(parent: Room) -> tuple[Table, list[AccompanyingItem]]:
		return Table("table", parent), []
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		return "on", {}
	
class Shelf(Container):
	MIN_LEVELS = 7
	MAX_LEVELS = 15
	LEVEL_PARAM = "?e"
	LEVEL_TYPE = "shelf_level"
	EXTRA_INFO: dict[str, Any] = {"level_token" : LEVEL_PARAM}

	@staticmethod
	def get_level_name(level: int) -> str:
		return "shelf_level_" + str(level)
	
	LEVEL_OBJECTS: list[Instance] = []
	for i in range(MAX_LEVELS):
		LEVEL_OBJECTS.append(Instance(EntityID(get_level_name.__func__(i + 1), LEVEL_TYPE), []))

	def __init__(self, name: str, parent: Room, levels: int) -> None:
		super().__init__(name, parent)
		self.levels = levels
	
	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return True
	
	@staticmethod
	def generate_instance(parent: Room) -> tuple[Shelf, list[AccompanyingItem]]:
		return Shelf("shelf", parent, random.randint(Shelf.MIN_LEVELS, Shelf.MAX_LEVELS)), []
	
	def get_description(self) -> str:
		items_by_level: dict[int, list[MovableItem]] = {level : [] for level in range(1, self.levels + 1)}
		for item in self.items:
			items_by_level[item.extra_location_info["level_num"]].append(item)
		description = f"The {self.name} has {self.levels} levels. "
		for level, item_list in items_by_level.items():
			if len(item_list) == 0:
				continue
			description += f"The {Shelf.integer_to_ordinal(level)} level of the {self.name} has {Shelf.get_item_list_description(item_list)}. "
		return description
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		level = random.randrange(self.levels) + 1
		return f"on the {Shelf.integer_to_ordinal(level)} level of", \
				{
					"level_num" : level,
					"level_token": self.get_level_name(level),
					"extra_attributes": [Attribute("on_shelf_level", Shelf.LEVEL_OBJECTS[level - 1].entity_id)]
				}

	@staticmethod
	def integer_to_ordinal(number):
		if number % 100 in [11, 12, 13]:
			return str(number) + "th"
		elif number % 10 == 1:
			return str(number) + "st"
		elif number % 10 == 2:
			return str(number) + "nd"
		elif number % 10 == 3:
			return str(number) + "rd"
		else:
			return str(number) + "th"
		
	@classmethod
	def get_pddl_domain_predicates(cls) -> list[Predicate]:
		predicates = super().get_pddl_domain_predicates()
		predicates.append(Predicate("shelf_has_level", [f"?a - {cls.get_type_name()}", f"?b - {cls.LEVEL_TYPE}"]))
		predicates.append(Predicate("on_shelf_level", [cls.get_holdable_param("?a"), f"?b - {cls.LEVEL_TYPE}"]))
		return predicates
	
	@classmethod
	def get_place_action(cls) -> Action:
		place = super().get_place_action()
		place.preconditions.append(f"shelf_has_level {super().CONTAINER_PARAM} {cls.LEVEL_PARAM}")
		return place
	
	@classmethod
	def get_default_param_list(cls) -> list[str]:
		param_list = super().get_default_param_list()
		param_list.append(f"{cls.LEVEL_PARAM} - {cls.LEVEL_TYPE}")
		return param_list
	
	@classmethod
	def get_contains_predicates(cls, container_param: str, item_param: str, **kwargs) -> list[str]:
		return super().get_contains_predicates(container_param, item_param) + [f"on_shelf_level {item_param} {kwargs['level_token']}"]
	
	@classmethod
	def get_required_types(cls) -> list[str]:
		types = super().get_required_types()
		types.append(cls.LEVEL_TYPE)
		return types
	
	def get_init_conditions(self) -> list[str]:
		conditions = super().get_init_conditions()
		for i in range(self.levels):
			conditions.append(f"shelf_has_level {self.token_name} {self.get_level_name(i + 1)}")
		return conditions
	
	def get_yaml_attributes(self) -> list[Attribute]:
		attributes = Container.get_yaml_attributes(self)
		for i in range(self.levels):
			attributes.append(Attribute("shelf_has_level", Shelf.LEVEL_OBJECTS[i].entity_id))
		return attributes
	
	@staticmethod
	def get_static_entities() -> list[Instance]:
		return Shelf.LEVEL_OBJECTS

class Pantry(Shelf):
	MIN_FOODS = 30
	MAX_FOODS = 40

	def __init__(self, name: str, parent: Room, levels: int, foods: list[NonPerishable]) -> None:
		super().__init__(name, parent, levels)
		self.foods = foods

	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return issubclass(item_type, NonPerishable)
		
	@classmethod
	def get_contains_relation(cls) -> str:
		return Shelf.get_contains_relation()

	@classmethod
	def get_place_action_name(cls) -> str:
		return Shelf.get_place_action_name()

	@classmethod
	def get_remove_action_name(cls) -> str:
		return Shelf.get_remove_action_name()
	
	@classmethod
	def get_pddl_domain_predicates(cls) -> list[Predicate]:
		return []
	
	@classmethod
	def get_place_action(cls) -> Action:
		return Shelf.get_place_action()
	
	@classmethod
	def get_remove_action(cls) -> Action:
		return Shelf.get_remove_action()
	
	@classmethod
	def get_pddl_domain_actions(cls) -> list[Action]:
		return []

	@staticmethod
	def generate_instance(parent: Room) -> tuple[Pantry, list[NonPerishable]]:
		foods: list[NonPerishable] = []
		food_item = NonPerishable.generate_instance()
		threshold = random.randint(Pantry.MIN_FOODS, Pantry.MAX_FOODS)
		while food_item is not None and len(foods) < threshold:
			assert isinstance(food_item, NonPerishable)
			foods.append(food_item)
			food_item = NonPerishable.generate_instance()
		return Pantry("pantry", parent, random.randint(Shelf.MIN_LEVELS, Shelf.MAX_LEVELS), foods), foods
	
	def generate_special_goal(self, agent: Agent, combined: bool = False) -> Goal:
		num_splits = random.randint(min(3, Shelf.MIN_LEVELS), min(len(NonPerishable.categories), Shelf.MAX_LEVELS))
		levels = random.sample(range(1, self.levels + 1), num_splits)
		categories = NonPerishable.categories.copy()
		random.shuffle(categories)
		category_to_level: dict[str, int] = {}
		split_size, split_remainder = divmod(len(categories), num_splits)
		goal_str = f"Organize the {self.name if combined else self.get_full_name_with_room()} as follows. "

		idx = 0
		for i, level in enumerate(levels):
			newidx = idx + split_size + (1 if i < split_remainder else 0)
			curr_categories = categories[idx : newidx]
			idx = newidx
			for category in curr_categories:
				category_to_level[category] = level
			
			categories_str = curr_categories[0]
			for j, category in enumerate(curr_categories[1:]):
				if j == len(curr_categories) - 2:
					categories_str += " and " + category
				else:
					categories_str += ", " + category

			goal_str += f"Place all {categories_str} on the {Shelf.integer_to_ordinal(level)} level. "

		predicates: list[str] = []
		agent.parent = self.parent
		for food in self.foods:
			if self != food.container:
				food.exchange_container(self)
			category = NonPerishable.item_to_category[re.sub(r"[^a-z]", "", food.name)]
			level = category_to_level[category]
			food.relative_location = f"on the {Shelf.integer_to_ordinal(level)} level of"
			food.extra_location_info = {
				"level_num": level,
				"level_token": self.get_level_name(level),
				"extra_attributes": [Attribute("on_shelf_level", Shelf.LEVEL_OBJECTS[level - 1].entity_id)]
			}
			predicates += self.get_contains_predicates(self.token_name, food.token_name, **food.extra_location_info)

		return Goal(goal_str, predicates)
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		if random.choice([True, False]):
			goal = super().generate_goal(people, all_items, agent)
			if goal is not None:
				return goal
		return self.generate_special_goal(agent)
	
	@classmethod
	def get_default_param_list(cls) -> list[str]:
		param_list = super().get_default_param_list()
		param_list.append(f"{cls.LEVEL_PARAM} - {cls.LEVEL_TYPE}")
		return param_list
	
	@classmethod
	def get_type_name(cls) -> str:
		return Shelf.get_type_name()

	@classmethod
	def get_required_types(cls) -> list[str]:
		return []
	
	@staticmethod
	def get_static_entities() -> list[Instance]:
		return []

class Fridge(Container):
	MIN_FOODS = 30
	MAX_FOODS = 40
	def __init__(self, name: str, parent: Room, foods: list[Perishable]) -> None:
		super().__init__(name, parent)
		self.foods = foods

	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return issubclass(item_type, Perishable)
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		return "inside", {}

	@staticmethod
	def generate_instance(parent: Room) -> tuple[Fridge, list[Perishable]]:
		foods: list[Perishable] = []
		food_item = Perishable.generate_instance()
		threshold = random.randint(Fridge.MIN_FOODS, Fridge.MAX_FOODS)
		while food_item is not None and len(foods) < threshold:
			assert isinstance(food_item, Perishable)
			foods.append(food_item)
			food_item = Perishable.generate_instance()
		return Fridge("fridge", parent, foods), foods
	
	def generate_special_goal(self, agent: Agent, combined: bool = False) -> Goal:
		predicates: list[str] = []
		agent.parent = self.parent
		for food in self.foods:
			if self != food.container:
				food.exchange_container(self)
			predicates += self.get_contains_predicates(self.token_name, food.token_name, **food.extra_location_info)
		return Goal(
			f"Move all fruits/vegetables, dairy products, and frozen food to the {self.name if combined else self.get_full_name_with_room()}.",
			predicates
		)
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		if random.choice([True, False]):
			goal = super().generate_goal(people, all_items, agent)
			if goal is not None:
				return goal
		return self.generate_special_goal(agent)

class Toilet(StationaryItem):
	@staticmethod
	def generate_instance(parent: Room) -> tuple[Toilet, list[AccompanyingItem]]:
		return Toilet("toilet", parent), []
	
	def perform_action(self, people: list[Person]) -> str | None:
		return None
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return []
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		return []
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		return None
	
	def get_description(self) -> str:
		return ""

class Sink(StationaryInteractable):
	FAUCET_ON_RELATION = "faucet_on"

	def __init__(self, name: str, parent: Room, faucet_on: bool) -> None:
		super().__init__(name, parent)
		self.faucet_on = faucet_on

	def generate_query_answer(self) -> tuple[str, str]:
		return f"Is the faucet of the {self.get_full_name_with_room()} on or off?", "The faucet is {}.".format("on" if self.faucet_on else "off")
	
	def perform_action(self, people: list[Person]) -> str | None:
		person = random.choice(people)
		person.parent = self.parent
		self.faucet_on = not self.faucet_on
		return "{} went to {} and turned {} the faucet of the {}.".format(person.name, self.parent.name, "on" if self.faucet_on else "off", self.name)
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [Predicate(Sink.FAUCET_ON_RELATION, ["?a - " + Sink.get_type_name()])]
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		param_list = ["?a - " + Sink.get_type_name(), "?b - " + Room.TYPE_NAME, "?c - " + Agent.TYPE_NAME]
		base_preconditions = [Room.get_in_room_predicate("?b", "?a"), Agent.get_in_room_predicate("?c", "?b")]
		return [
			Action(
				"turn_on_faucet",
				param_list,
				base_preconditions + [f"not ({Sink.FAUCET_ON_RELATION} ?a)"],
				[f"{Sink.FAUCET_ON_RELATION} ?a"]
			),
			Action(
				"turn_off_faucet",
				param_list,
				base_preconditions + [f"{Sink.FAUCET_ON_RELATION} ?a"],
				[f"not ({Sink.FAUCET_ON_RELATION} ?a)"]
			)
		]
	
	def get_special_init_conditions(self) -> list[str]:
		if self.faucet_on:
			return [f"{Sink.FAUCET_ON_RELATION} {self.token_name}"]
		return []
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		return [Attribute(Sink.FAUCET_ON_RELATION, self.faucet_on)]
	
	@staticmethod
	def generate_instance(parent: Room) -> tuple[Sink, list[AccompanyingItem]]:
		return Sink("sink", parent, random.choice([True, False])), []
	
	def get_description(self) -> str:
		return "The sink has a faucet that can be turned on and off. It is currently {}. ".format("on" if self.faucet_on else "off")
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		self.faucet_on = random.choice([True, False])
		pred = f"{Sink.FAUCET_ON_RELATION} {self.token_name}"
		agent.parent = self.parent
		return Goal(
			"Make sure that the faucet of the {} is {}.".format(self.get_full_name_with_room(), "on" if self.faucet_on else "off"),
			[pred if self.faucet_on else f"not ({pred})"]
		)

class KitchenSink(InteractableContainer):
	MIN_DISHES = 7
	MAX_DISHES = 10
	def __init__(self, name: str, parent: Room, faucet_on: bool, dishes: list[Kitchenware | LiquidContainer]) -> None:
		super().__init__(name, parent)
		self.faucet_on = faucet_on
		self.dishes = dishes
		for dish in self.dishes:
			dish.sink = self

	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return issubclass(item_type, Kitchenware) or issubclass(item_type, LiquidContainer)
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		return "in", {}

	def interact(self, people: list[Person]) -> str | None:
		self.faucet_on = not self.faucet_on
		person = random.choice(people)
		person.parent = self.parent
		return "{} went to {} and turned {} the faucet of the {}.".format(person.name, self.parent.name, "on" if self.faucet_on else "off", self.name)
	
	def generate_query_answer(self) -> tuple[str, str]:
		return f"Is the faucet of the {self.get_full_name_with_room()} on or off?", "The faucet is {}.".format("on" if self.faucet_on else "off")

	def get_special_init_conditions(self) -> list[str]:
		special_conds = []
		for dish in self.dishes:
			if dish.clean:
				special_conds.append(f"dish_is_clean {dish.token_name}")
		if self.faucet_on:
			return [f"{Sink.FAUCET_ON_RELATION} {self.token_name}"]
		return []
	
	def get_interactable_description(self) -> str:
		return "The sink has a faucet that can be turned on and off. It is currently {}. ".format("on" if self.faucet_on else "off")
	
	@staticmethod
	def get_special_domain_predicates() -> list[Predicate]:
		return [Predicate("dish_is_clean", [f"?a - {Kitchenware.get_type_name()}"])]
	
	@staticmethod
	def get_special_domain_actions() -> list[Action]:
		return [
			Action(
				"wash",
				[f"?a - {Kitchenware.get_type_name()}", f"?b - {KitchenSink.get_type_name()}", "?c - " + Room.TYPE_NAME, "?d - " + Agent.TYPE_NAME],
				[Room.get_in_room_predicate("?c", "?b"), Agent.get_in_room_predicate("?d", "?c")] + KitchenSink.get_contains_predicates("?b", "?a"),
				["dish_is_clean ?a"]
			)
		]
	
	@classmethod
	def get_required_types(cls) -> list[str]:
		return [f"{cls.get_type_name()} - {Sink.get_type_name()}"]
	
	@staticmethod
	def generate_instance(parent: Room) -> tuple[KitchenSink, list[Kitchenware | LiquidContainer]]:
		dishes: list[Kitchenware | LiquidContainer] = []
		dish = Kitchenware.generate_instance()
		threshold = random.randint(KitchenSink.MIN_DISHES, KitchenSink.MAX_DISHES)
		while dish is not None and len(dishes) < threshold:
			dishes.append(cast(Kitchenware, dish))
			dish = Kitchenware.generate_instance()
		glass = LiquidContainer.generate_instance()
		if glass is not None:
			dishes.append(glass)
		return KitchenSink("sink", parent, random.choice([True, False]), dishes), dishes
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		return [Attribute(Sink.FAUCET_ON_RELATION, self.faucet_on)]
	
	def generate_interactable_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		# 1/3 regular Sink goal, 1/3 clean goal, 1/3 return goal
		if random.choice([True, False, False]):
			goal = Sink.generate_goal(self, people, all_items, agent) # type: ignore
			if goal is not None:
				return goal
		clean_goal = random.choice([True, False])
		predicates: list[str] = []
		agent.parent = self.parent
		for dish in self.dishes:
			if self != dish.container:
				dish.exchange_container(self)
			if clean_goal:
				dish.clean = True
				predicates.append(f"dish_is_clean {dish.token_name}")
			else:
				predicates += self.get_contains_predicates(self.token_name, dish.token_name, **dish.extra_location_info)
		if clean_goal:
			return Goal("Please wash all the dishes.", predicates)
		return Goal(
			f"Please return all dishes to the {self.get_full_name_with_room()}.",
			predicates
		)

class Washer(Container):
	def __init__(self, name: str, parent: Room) -> None:
		super().__init__(name, parent)
	
	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return item_type in [Cloth]
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		return "in", {}

	@staticmethod
	def generate_instance(parent: Room) -> tuple[Washer, list[AccompanyingItem]]:
		return Washer("washer", parent), []
	
	@classmethod
	def get_pddl_domain_actions(cls) -> list[Action]:
		actions = super().get_pddl_domain_actions()
		cloth_preconditions = ") (".join(cls.get_contains_predicates("?a", "?d", **cls.EXTRA_INFO))
		actions.append(Action(
			"run_washer_cycle",
			[f"?a - {cls.get_type_name()}", f"?b - {Room.TYPE_NAME}", f"?c - {Agent.TYPE_NAME}"],
			[Room.get_in_room_predicate("?b", "?a"), Agent.get_in_room_predicate("?c", "?b")],
			[f"forall (?d - {Cloth.get_type_name()}) (when ({cloth_preconditions}) (and (cloth_is_clean ?d) (not (cloth_is_dry ?d))))"]))
		return actions

class Dryer(Container):
	def __init__(self, name: str, parent: Room) -> None:
		super().__init__(name, parent)
	
	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return item_type in [Cloth]
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		return "in", {}

	@staticmethod
	def generate_instance(parent: Room) -> tuple[Dryer, list[AccompanyingItem]]:
		return Dryer("dryer", parent), []
	
	@classmethod
	def get_pddl_domain_actions(cls) -> list[Action]:
		actions = super().get_pddl_domain_actions()
		cloth_preconditions = ") (".join(cls.get_contains_predicates("?a", "?d", **cls.EXTRA_INFO))
		actions.append(Action(
			"run_dryer_cycle",
			[f"?a - {cls.get_type_name()}", f"?b - {Room.TYPE_NAME}", f"?c - {Agent.TYPE_NAME}"],
			[Room.get_in_room_predicate("?b", "?a"), Agent.get_in_room_predicate("?c", "?b")],
			[f"forall (?d - {Cloth.get_type_name()}) (when ({cloth_preconditions}) (cloth_is_dry ?d))"]))
		return actions

class LaundryBasket(Container):
	def __init__(self, name: str, parent: Room) -> None:
		super().__init__(name, parent)
	
	@staticmethod
	def can_hold(item_type: type[MovableItem]) -> bool:
		return item_type in [Cloth]
	
	def generate_relative_location(self) -> tuple[str, dict[Any, Any]]:
		return "in", {}

	@staticmethod
	def generate_instance(parent: Room) -> tuple[LaundryBasket, list[AccompanyingItem]]:
		return LaundryBasket("laundry basket", parent), []

class Book(MovableItem):
	with open(os.path.join(DIR, "items/book_titles.txt")) as f:
		available_titles = f.read().splitlines()

	def __init__(self, title: str) -> None:
		prefix = re.sub(r"[^a-zA-Z0-9]+", "_", title).lower()
		super().__init__(f'book called "{title}"', prefix + "_book", f'"{title}" book')

	@staticmethod
	def generate_instance() -> Book | None:
		if len(Book.available_titles) == 0:
			return None
		idx = random.randrange(len(Book.available_titles))
		return Book(Book.available_titles.pop(idx))

class Pen(MovableItem):
	with open(os.path.join(DIR, "items/colors.txt")) as f:	
		available_colors = f.read().lower().splitlines()

	def __init__(self, color: str) -> None:
		super().__init__(f"{color} pen", color + "_pen", f"{color} pen")
	
	@staticmethod
	def generate_instance() -> Pen | None:
		if len(Pen.available_colors) == 0:
			return None
		idx = random.randrange(len(Pen.available_colors))
		return Pen(Pen.available_colors.pop(idx))

class Singleton(MovableItem):
	def __init__(self, name: str) -> None:
		token_name = re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()
		super().__init__(name, token_name, name)
	
	@staticmethod
	@abstractmethod
	def get_available_names() -> list[str]:
		pass

	@classmethod
	def generate_instance(cls) -> Singleton | None:
		names = cls.get_available_names()
		if len(names) == 0:
			return None
		return cls(names.pop(random.randrange(len(names))))

class Perishable(Singleton, AccompanyingItem):
	with open(os.path.join(DIR, "items/perishable_foods.txt")) as f:
		available_foods = f.read().lower().splitlines()
	
	@staticmethod
	def get_available_names() -> list[str]:
		return Perishable.available_foods
	
	@classmethod
	def get_required_types(cls) -> list[str]:
		return ["food", f"{cls.get_type_name()} - food"]
	
class NonPerishable(Singleton, AccompanyingItem):
	available_foods: list[str] = []
	categories: list[str] = []
	item_to_category: dict[str, str] = {}	
	with open(os.path.join(DIR, "items/nonperishable_foods.txt")) as f:
		foods = f.read().lower().strip().split("\n\n")
	for category in foods:
		items = category.splitlines()
		name, items = items[0], items[1:]
		available_foods += items
		categories.append(name)
		for item in items:
			item = re.sub(r"[^a-z]", "", item)
			item_to_category[item] = name
	del foods, category, items, name, item

	@staticmethod
	def get_available_names() -> list[str]:
		return NonPerishable.available_foods
	
	@classmethod
	def get_required_types(cls) -> list[str]:
		return [f"{cls.get_type_name()} - food"]

class Kitchenware(Singleton, AccompanyingItem):
	available_kitchenware = ["plate", "bowl", "fork", "spoon", "knife", "frying pan", "pot", "ladle", "whisk"]

	@staticmethod
	def get_available_names() -> list[str]:
		return Kitchenware.available_kitchenware
	
	def perform_action(self, people: list[Person]) -> str | None:
		if self.clean:
			self.clean = False
			return f"Something spilled on {self.shortened_name} so it got dirty."
		else:
			self.clean = True
			person = random.choice(people)
			person.parent = self.sink.parent
			self.exchange_container(self.sink)
			return f"{person.name} went to {person.parent.name} and washed {self.shortened_name} in the {self.sink.name}."
	
	def __init__(self, name: str) -> None:
		super().__init__(name)
		self.clean = random.choice([True, False])
		self.sink: KitchenSink
	
	def get_yaml_attributes(self) -> list[Attribute]:
		return super().get_yaml_attributes() + [Attribute("dish_is_clean", self.clean)]

class Window(StationaryInteractable):
	def __init__(self, parent: Room, open: bool) -> None:
		super().__init__("window", parent)
		self.open = open
	
	def generate_query_answer(self) -> tuple[str, str]:
		return f"Are the blinds of the {self.get_full_name_with_room()} open or closed?", "The window blinds are {}.".format("open" if self.open else "closed")
	
	def perform_action(self, people: list[Person]) -> str | None:
		self.open = not self.open
		person = random.choice(people)
		person.parent = self.parent
		return "{} went to {} and {} the blinds of the {}.".format(person.name, self.parent.name, "opened" if self.open else "closed", self.name)
	
	def get_description(self) -> str:
		return "The window has blinds that can open and close. They are currently {}. ".format("open" if self.open else "closed")

	@staticmethod
	def generate_instance(parent: Room) -> tuple[Window, list[AccompanyingItem]]:
		return Window(parent, random.choice([True, False])), []
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [Predicate("window_open", ["?a - window"])]

	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		param_list = ["?a - window", "?b - " + Room.TYPE_NAME, "?c - " + Agent.TYPE_NAME]
		base_preconditions = [Room.get_in_room_predicate("?b", "?a"), Agent.get_in_room_predicate("?c", "?b")]
		return [
			Action(
				"open_window",
				param_list,
				base_preconditions + ["not (window_open ?a)"],
				["window_open ?a"]
			),
			Action(
				"close_window",
				param_list,
				base_preconditions + ["window_open ?a"],
				["not (window_open ?a)"]
			)
		]
	
	def get_special_init_conditions(self) -> list[str]:
		if self.open:
			return ["window_open " + self.token_name]
		return []
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		return [Attribute("window_open", self.open)]
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		self.open = random.choice([True, False])
		pred = f"window_open {self.token_name}"
		agent.parent = self.parent
		return Goal(
			"Make sure the blinds of the {} are {}.".format(self.get_full_name_with_room(), "open" if self.open else "closed"),
			[pred if self.open else f"not ({pred})"]
		)

class Light(StationaryInteractable):
	def __init__(self, name: str, parent: Room, on: bool) -> None:
		super().__init__(name, parent)
		self.on = on
	
	def generate_query_answer(self) -> tuple[str, str]:
		return f"Is the {self.get_full_name_with_room()} on or off?", "The light is {}.".format("on" if self.on else "off")
	
	def perform_action(self, people: list[Person]) -> str | None:
		self.on = not self.on
		person = random.choice(people)
		person.parent = self.parent
		return "{} went to {} and turned {} the {}.".format(person.name, self.parent.name, "on" if self.on else "off", self.name)
	
	def get_description(self) -> str:
		return "The light turns on and off. It is currently {}. ".format("on" if self.on else "off")
	
	@staticmethod
	def generate_instance(parent: Room) -> tuple[Light, list[AccompanyingItem]]:
		return Light("overhead light", parent, random.choice([True, False])), []
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [Predicate("light_on", ["?a - " + Light.get_type_name()])]
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		param_list = ["?a - " + Light.get_type_name(), "?b - " + Room.TYPE_NAME, "?c - " + Agent.TYPE_NAME]
		base_preconditions = [Room.get_in_room_predicate("?b", "?a"), Agent.get_in_room_predicate("?c", "?b")]
		return [
			Action(
				"turn_on_light",
				param_list,
				base_preconditions + ["not (light_on ?a)"],
				["light_on ?a"]
			),
			Action(
				"turn_off_light",
				param_list,
				base_preconditions + ["light_on ?a"],
				["not (light_on ?a)"]
			)
		]
	
	def get_special_init_conditions(self) -> list[str]:
		if self.on:
			return ["light_on " + self.token_name]
		return []
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		return [Attribute("light_on", self.on)]
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		self.on = random.choice([True, False])
		pred = f"light_on {self.token_name}"
		agent.parent = self.parent
		return Goal(
			"Make sure the {} is {}.".format(self.get_full_name_with_room(), "on" if self.on else "off"),
			[pred if self.on else f"not ({pred})"]
		)

class Remote(AccompanyingItem):
	def __init__(self, name: str) -> None:
		token_name = re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()
		super().__init__(name, token_name, name, True)

	@staticmethod
	def generate_instance() -> Remote | None:
		return Remote("remote")

class Cloth(MovableInteractable):
	with open(os.path.join(DIR, "items/clothes.txt")) as f:	
		available_clothes = f.read().splitlines()
	with open(os.path.join(DIR, "items/names.txt")) as f:	
		available_names = f.read().splitlines()
	used_combos = set()

	def __init__(self, type: str, owner: str, clean: bool) -> None:
		type_token = re.sub(r"[^a-zA-Z0-9]+", "_", type).lower()
		super().__init__(f"{type} that belongs to {owner}", f"{owner.lower()}_{type_token}", f"{owner}'s {type}", False)
		self.clean = clean
	
	@staticmethod
	def generate_instance() -> Cloth | None:
		for _ in range(MAX_ITER):
			person = random.choice(Cloth.available_names)
			cloth = random.choice(Cloth.available_clothes)
			if (person, cloth) not in Cloth.used_combos:
				Cloth.used_combos.add((person, cloth))
				return Cloth(cloth, person, random.choice([True, False]))
		return None
	
	def get_special_init_conditions(self) -> list[str]:
		conditions = ["cloth_is_dry " + self.token_name]
		if self.clean:
			conditions.append("cloth_is_clean " + self.token_name)
		return conditions
	
	def generate_interactable_qa(self) -> tuple[str, str]:
		return f"Is {self.shortened_name} clean?", "Yes." if self.clean else "No."
	
	def interact(self, people: list[Person]) -> str | None:
		if not self.clean:
			return None
		self.clean = False
		return f"{random.choice(people).name} accidentally spilled something on {self.shortened_name} so now it's dirty."
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [
			Predicate("cloth_is_clean", ["?a - " + Cloth.get_type_name()]),
			Predicate("cloth_is_dry", ["?a - " + Cloth.get_type_name()])
		]
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		return []
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		return [Attribute("cloth_is_clean", self.clean), Attribute("cloth_is_dry", True)]

class TV(StationaryInteractable):
	class Channel:
		TYPE_NAME = "channel"
		def __init__(self, name: str) -> None:
			self.name = name
			self.token_name = re.sub(r"[^a-zA-Z0-9]+", "_", name).lower()
			self.entity_id = EntityID(self.token_name, "channel")

	CHANNELS = [
		Channel("the Discovery Channel"),
		Channel("Cartoon Network"),
		Channel("NBC"),
		Channel("CNN"),
		Channel("Fox News"),
		Channel("ESPN")
	]
	CHANNEL_OBJECTS = [Instance(channel.entity_id, []) for channel in CHANNELS]

	def __init__(self, parent: Room, on: bool, curr_channel: Channel, remote: Remote) -> None:
		super().__init__("TV", parent)
		self.on = on
		self.curr_channel = curr_channel
		self.remote = remote
		remote.name = f"remote for {parent.name} TV"
		remote.set_shortened_name(remote.name, True)
		self.remote.token_name = self.token_name + "_remote"
		self.remote.initialize_entity_id()
	
	def generate_query_answer(self) -> tuple[str, str]:
		query = f"Is the TV in {self.parent.name} on or off? If it's on, what channel is it playing?"
		answer = "The TV is {}{}.".format("on" if self.on else "off", f" and is playing {self.curr_channel.name}" if self.on else "")
		return query, answer
	
	def perform_action(self, people: list[Person]) -> str | None:
		if not isinstance(self.remote.container, Person):
			return None
		if self.on:
			# keep the TV on
			if random.choice([True, False]):
				self.curr_channel = random.choice(TV.CHANNELS)
				return f"{self.remote.container.name} switched the channel of the TV in {self.parent.name} to {self.curr_channel.name}."
			# turn the TV off
			self.on = False
			return f"{self.remote.container.name} turned off the TV in {self.parent.name}."
		self.on = True
		self.curr_channel = random.choice(TV.CHANNELS)
		return f"{self.remote.container.name} turned on the TV in {self.parent.name} and set it to {self.curr_channel.name}."
	
	def get_description(self) -> str:
		if self.on:
			return f"The TV is currently on and is playing {self.curr_channel.name}. "
		return "The TV is currently off. "
	
	@staticmethod
	def generate_instance(parent: Room) -> tuple[TV, list[AccompanyingItem]]:
		remote = Remote.generate_instance()
		assert remote is not None
		return TV(parent, random.choice([True, False]), random.choice(TV.CHANNELS), remote), [remote]
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [Predicate("tv_on", ["?a - tv"]), Predicate("tv_playing_channel", ["?a - tv", "?b - channel"])]
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		return [
			Action("turn_tv_on", ["?a - tv", "?b - channel"], ["not (tv_on ?a)"], ["tv_on ?a", "tv_playing_channel ?a ?b"]),
			Action("turn_tv_off", ["?a - tv", "?b - channel"], ["tv_on ?a", "tv_playing_channel ?a ?b"], ["not (tv_on ?a)", "not (tv_playing_channel ?a ?b)"]),
			Action("switch_tv_channel", ["?a - tv", "?b - channel", "?c - channel"], ["tv_playing_channel ?a ?b"], ["tv_playing_channel ?a ?c", "not (tv_playing_channel ?a ?b)"])
		]
	
	@classmethod
	def get_required_types(cls) -> list[str]:
		types = super().get_required_types()
		types.append(cls.Channel.TYPE_NAME)
		return types
	
	def get_special_init_conditions(self) -> list[str]:
		if self.on:
			return ["tv_on " + self.token_name, f"tv_playing_channel {self.token_name} {self.curr_channel.token_name}"]
		return []
	
	@staticmethod
	def get_static_entities() -> list[Instance]:
		return TV.CHANNEL_OBJECTS
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		attributes = [Attribute("tv_on", self.on)]
		if self.on:
			attributes.append(Attribute("tv_playing_channel", self.curr_channel.entity_id))
		return attributes
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		usable_people = people.copy()
		person = usable_people.pop(random.randrange(len(usable_people)))
		while self.remote not in person.items:
			if len(usable_people) == 0:
				return None
			person = usable_people.pop(random.randrange(len(usable_people)))
		agent.parent = person.parent
		self.remote.exchange_container(person)
		return Goal(
			f"{person.name} is trying to use the TV in {self.parent.name} but they need the remote. Please hand it to them.",
			[person.get_in_hand_predicate(self.remote.token_name, person.token_name)]
		)

class Phone(MovableInteractable):
	with open(os.path.join(DIR, "items/names.txt")) as f:	
		available_names = f.read().splitlines()

	def __init__(self, owner: str) -> None:
		super().__init__(f"phone that belongs to {owner}", owner.lower() + "_phone", f"{owner}'s phone", use_default_article=False)
		self.ringing = False
	
	def get_special_init_conditions(self) -> list[str]:
		if self.ringing:
			return ["phone_ringing " + self.token_name]
		return []
	
	def generate_interactable_qa(self) -> tuple[str, str]:
		return f"Is {self.shortened_name} ringing?", "Yes." if self.ringing else "No."
	
	def interact(self, people: list[Person]) -> str | None:
		self.ringing = not self.ringing
		return "{} {} ringing.".format(self.shortened_name, "started" if self.ringing else "stopped")
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [Predicate("phone_ringing", ["?a - " + Phone.get_type_name()])]
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		return [Action("answer_phone", ["?a - " + Phone.get_type_name()], ["phone_ringing ?a"], ["not (phone_ringing ?a)"])]

	@staticmethod
	def generate_instance() -> Phone | None:
		if len(Phone.available_names) == 0:
			return None
		return Phone(Phone.available_names.pop(random.randrange(len(Phone.available_names))))
	
	def get_special_yaml_attributes(self) -> list[Attribute]:
		return [Attribute("phone_ringing", self.ringing)]
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		if random.choice([True, False]):
			goal = super().generate_goal(people, all_items, agent)
			if goal is not None:
				return goal
		if self.ringing:
			self.ringing = False
			return Goal(f"Answer {self.shortened_name}.", [f"not (phone_ringing {self.token_name})"])
		return None

class LiquidContainer(MovableInteractable, AccompanyingItem):
	generated = False

	LIQUIDS: list[Instance] = []
	for l in ["water", "juice", "coffee", "soda"]:
		LIQUIDS.append(Instance(EntityID(l, "liquid"), []))
	
	def __init__(self) -> None:
		super().__init__(f"glass", "glass", "the glass")
		self.empty = random.choice([True, False])
		if self.empty:
			self.liquid = None
			self.clean = random.choice([True, False])
		else:
			self.liquid = random.choice(LiquidContainer.LIQUIDS)
			self.clean = False
		self.sink: KitchenSink
	
	def get_special_init_conditions(self) -> list[str]:
		if self.empty:
			return ["glass_empty " + self.token_name]
		assert(isinstance(self.liquid, Instance))
		return [f"glass_has_liquid {self.token_name} {self.liquid.entity_id.name}"]
	
	def generate_interactable_qa(self) -> tuple[str, str]:
		liquid = ""
		if not self.empty:
			assert(isinstance(self.liquid, Instance))
			liquid = self.liquid.entity_id.name
		return f"Is {self.shortened_name} empty? If not, what does it contain?", "It is empty." if self.empty else f"It is not empty. It contains {liquid}."
	
	def interact(self, people: list[Person]) -> str | None:
		if random.choice([True, False]):
			interaction = Kitchenware.perform_action(self, people) # type: ignore
			if interaction is not None:
				return interaction
		if self.empty:
			self.empty = False
			self.liquid = random.choice(LiquidContainer.LIQUIDS)
			return f"{random.choice(people).name} filled {self.shortened_name} with {self.liquid.entity_id.name}."
		self.empty = True
		self.liquid = None
		return f"{random.choice(people).name} emptied {self.shortened_name}."
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [
			Predicate("glass_empty", ["?a - " + LiquidContainer.get_type_name()]),
			Predicate("glass_has_liquid", ["?a - " + LiquidContainer.get_type_name(), "?b - liquid"])
		  ]
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		return [
			Action(
				"empty_glass",
				["?a - " + LiquidContainer.get_type_name(), "?b - liquid", "?c - " + Agent.TYPE_NAME],
				[Agent.get_in_hand_predicate("?a", "?c"),  "glass_has_liquid ?a ?b"],
				["glass_empty ?a", "not (glass_has_liquid ?a ?b)"]
			),
			Action(
				"fill_with_liquid",
				["?a - " + LiquidContainer.get_type_name(), "?b - liquid", "?c - " + Agent.TYPE_NAME],
				[Agent.get_in_hand_predicate("?a", "?c"), "glass_empty ?a", "dish_is_clean ?a"],
				["not (glass_empty ?a)", "glass_has_liquid ?a ?b", "not (dish_is_clean ?a)"]
			)
		]

	@staticmethod
	def generate_instance() -> LiquidContainer | None:
		if LiquidContainer.generated:
			return None
		LiquidContainer.generated = True
		return LiquidContainer()
		
	def get_special_yaml_attributes(self) -> list[Attribute]:
		attributes = [Attribute("glass_empty", self.empty), Attribute("dish_is_clean", self.clean)]
		if not self.empty:
			assert(isinstance(self.liquid, Instance))
			attributes.append(Attribute("glass_has_liquid", self.liquid.entity_id))
		return attributes
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		if random.choice([True, False]):
			self.empty = True
			self.liquid = None
			return Goal("Make sure the glass is empty.", ["glass_empty " + self.token_name])
		
		self.empty = False
		self.liquid = random.choice(LiquidContainer.LIQUIDS)
		person = random.choice(people)
		agent.parent = person.parent
		self.exchange_container(person)
		return Goal(
			f"Hand {person.name} a glass of {self.liquid.entity_id.name}.",
			[person.get_in_hand_predicate(self.token_name, person.token_name), f"glass_has_liquid {self.token_name} {self.liquid.entity_id.name}"]
		)
	
	@staticmethod
	def get_static_entities() -> list[Instance]:
		return LiquidContainer.LIQUIDS
	
	@classmethod
	def get_required_types(cls) -> list[str]:
		return [f"{cls.get_type_name()} - {Kitchenware.get_type_name()}", "liquid"]

class Person:
	TYPE_NAME = "person"
	IN_HAND_RELATION = "in_person_hand"
	IN_ROOM_RELATION = "person_in_room"

	with open(os.path.join(DIR, "items/names.txt")) as f:	
		available_names = f.read().splitlines()
	
	def __init__(self, name: str, parent: Room) -> None:
		self.items: list[MovableItem] = []
		self.name = name
		self.token_name = name.lower()
		self.entity_id = EntityID(self.token_name, Person.TYPE_NAME)
		self.parent = parent

	@staticmethod
	def generate_person(rooms: list[Room]) -> Person | None:
		if len(Person.available_names) == 0:
			return None
		return Person(Person.available_names.pop(random.randrange(len(Person.available_names))), random.choice(rooms))
	
	@staticmethod
	def get_in_hand_predicate(item_param: str, person_param: str) -> str:
		return f"{Person.IN_HAND_RELATION} {item_param} {person_param}"
	
	@staticmethod
	def get_in_room_predicate(person_param: str, room_param: str) -> str:
		return f"{Person.IN_ROOM_RELATION} {person_param} {room_param}"
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		return [
			Predicate(Person.IN_HAND_RELATION, ["?a - (either {})".format(" ".join([movable_type.get_type_name() for movable_type in movable_types])), f"?b - {Person.TYPE_NAME}"]),
			Predicate(Person.IN_ROOM_RELATION, [f"?a - {Person.TYPE_NAME}", f"?b - {Room.TYPE_NAME}"]),
		]
	
	def get_pddl_objects(self) -> list[str]:
		return [self.token_name + " - " + self.TYPE_NAME]
	
	def get_init_conditions(self) -> list[str]:
		return [self.get_in_room_predicate(self.token_name, self.parent.token_name)]
	
	def get_yaml_instance(self) -> Instance:
		return Instance(self.entity_id, [Attribute(Person.IN_ROOM_RELATION, self.parent.entity_id)])
	
	def generate_goal(self, all_items: list[MovableItem], agent: Agent) -> Goal | None:
		return None
	
	def perform_action(self, all_items: list[MovableItem]) -> str | None:
		if len(self.items) >= 3:
			return None
		random.shuffle(all_items)
		for item in all_items:
			if item in self.items:
				continue
			action = f"{self.name} went to {item.container.parent.name} and picked up {item.shortened_name}." if isinstance(item.container, Container) \
						else f"{self.name} went to {item.container.parent.name} and took {item.shortened_name} from {item.container.name}."
			self.parent = item.container.parent
			item.exchange_container(self)
			return action
	
	def get_description(self) -> str:
		return f"{self.name} is in {self.parent.name}. "

class CollectiveGoal(ABC):
	@staticmethod
	@abstractmethod
	def generate_goal(movable_items: list[MovableItem], stationary_items: list[StationaryItem], agent: Agent) -> Goal | None:
		pass

class AnswerPhones(CollectiveGoal):
	@staticmethod
	def generate_goal(movable_items: list[MovableItem], stationary_items: list[StationaryItem], agent: Agent) -> Goal | None:
		predicate_list: list[str] = []
		for item in movable_items:
			if isinstance(item, Phone):
				item.ringing = False
				predicate_list.append(f"not (phone_ringing {item.token_name})")
		return Goal("Can you silence all the phones?", predicate_list)

class TurnOffAppliances(CollectiveGoal):
	@staticmethod
	def generate_goal(movable_items: list[MovableItem], stationary_items: list[StationaryItem], agent: Agent) -> Goal | None:
		predicate_list: list[str] = []
		last: StationaryItem | None = None
		for item in stationary_items:
			if isinstance(item, Light):
				item.on = False
				predicate_list.append(f"not (light_on {item.token_name})")
				last = item
			elif isinstance(item, Sink) or isinstance(item, KitchenSink):
				item.faucet_on = False
				predicate_list.append(f"not (faucet_on {item.token_name})")
				last = item
			elif isinstance(item, TV):
				item.on = False
				predicate_list.append(f"not (tv_on {item.token_name})")
				last = item
		if last:
			agent.parent = last.parent
		return Goal("The water and electricity bills are high. Can you turn off all appliances (other than the fridge)?", predicate_list)

class CleanAndDryClothes(CollectiveGoal):
	@staticmethod
	def generate_goal(movable_items: list[MovableItem], stationary_items: list[StationaryItem], agent: Agent) -> Goal | None:
		clothes = [item for item in movable_items if isinstance(item, Cloth)]
		laundry_basket = next(item for item in stationary_items if isinstance(item, LaundryBasket))
		if len(clothes) == 0:
			return None
		predicates = []
		agent.parent = laundry_basket.parent
		for cloth in clothes:
			cloth.exchange_container(laundry_basket)
			cloth.clean = True
			predicates += [f"cloth_is_clean {cloth.token_name}", f"cloth_is_dry {cloth.token_name}"] \
							+ laundry_basket.get_contains_predicates(laundry_basket.token_name, cloth.token_name)
		return Goal("Please wash and dry all the clothes, then place them in the laundry basket.", predicates)

item_types: list[type[RoomItem]]
movable_types: list[type[MovableItem]]
stationary_types: list[type[StationaryItem]]
collective_goal_types: list[type[CollectiveGoal]]

def get_items_and_probabilities(item_freq: dict[T, int], item_type_freq: dict[type[T], int], item_list: list[T]) -> tuple[list[T], list[float]]:
	item_probs = [1 / (item_freq.get(item, 0) + 1) ** 2 for item in item_list]
	item_type_probs = [1 / (item_type_freq.get(type(item), 0) + 1) ** 2 for item in item_list]
	item_probs = normalize_probabilities(item_probs)
	item_type_probs = normalize_probabilities(item_type_probs)
	return item_list.copy(), [item_prob * item_type_prob for item_prob, item_type_prob in zip(item_probs, item_type_probs)]

def normalize_probabilities(p: list[float]) -> list[float]:
	total = sum(p)
	return [x / total for x in p]

class Room(ABC):
	ROOM_PARAM = "?a"
	ITEM_PARAM = "?b"
	TYPE_NAME = "room"
	IN_ROOM_RELATION = "room_has"
	item_type_goal_freq: dict[type[StationaryItem], int] = {}
	item_type_update_freq: dict[type[StationaryItem], int] = {}

	def __init__(self, name: str, token_name: str) -> None:
		self.name = name
		self.token_name = token_name
		self.entity_id = EntityID(token_name, "room")
		self.items: list[StationaryItem] = []
		self.queryable_items: list[Queryable] = []
		self.yaml_instance: Instance
		self.item_goal_freq: dict[StationaryItem, int] = {}
		self.item_update_freq: dict[StationaryItem, int] = {}
	
	def add_item(self, item: StationaryItem) -> None:
		self.items.append(item)
		if isinstance(item, Queryable):
			self.queryable_items.append(item)
	
	@staticmethod
	@abstractmethod
	def generate_empty() -> Room | None:
		pass

	@classmethod
	def generate_outline(cls) -> tuple[Room, list[AccompanyingItem]] | None:
		room = cls.generate_empty()
		if room is None:
			return room

		attributes: list[Attribute] = []
		accompanying_items: list[AccompanyingItem] = []
		for item_type in stationary_types:
			if not cls.can_hold(item_type):
				continue
			item, additional = item_type.generate_instance(room)
			room.add_item(item)
			accompanying_items += additional
			attributes.append(Attribute(Room.IN_ROOM_RELATION, item.entity_id))
					
		room.yaml_instance = Instance(room.entity_id, attributes)
		return room, accompanying_items

	def populate(self, movable_items: list[MovableItem]) -> None:
		random.shuffle(self.items)
		for item in self.items:
			if isinstance(item, Container):
				item.populate(movable_items, max_allowed=random.randint(3, 7))
	
	def get_description(self) -> str:
		room_description = ""
		for i, item in enumerate(self.items):
			room_description += "{}{} has a{} {}. ".format(self.name.capitalize(), "" if i == 0 else " also", "n" if item.name[0] in "aeiou" else "", item.name)
			room_description += item.get_description()
		return room_description
	
	def perform_action(self, people: list[Person]) -> str | None:
		usable_items, probabilities = get_items_and_probabilities(self.item_update_freq, Room.item_type_update_freq, self.items)
		while len(usable_items) > 0:
			probabilities = normalize_probabilities(probabilities)
			idx = np.random.choice(np.arange(len(usable_items)), p=probabilities)
			item = usable_items.pop(idx)
			probabilities.pop(idx)

			action = item.perform_action(people)
			if action is not None:
				self.item_update_freq[item] = self.item_update_freq.get(item, 0) + 1
				Room.item_type_update_freq[type(item)] = Room.item_type_update_freq.get(type(item), 0) + 1
				return action
		return None
	
	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		usable_items, probabilities = get_items_and_probabilities(self.item_goal_freq, Room.item_type_goal_freq, self.items)
		while len(usable_items) > 0:
			probabilities = normalize_probabilities(probabilities)
			idx = np.random.choice(np.arange(len(usable_items)), p=probabilities)
			item = usable_items.pop(idx)
			probabilities.pop(idx)

			goal = item.generate_goal(people, all_items, agent)
			if goal is not None:
				self.item_goal_freq[item] = self.item_goal_freq.get(item, 0) + 1
				Room.item_type_goal_freq[type(item)] = Room.item_type_goal_freq.get(type(item), 0) + 1
				return goal
		return None
	
	def generate_query_answer(self) -> tuple[str, str]:
		return random.choice(self.queryable_items).generate_query_answer()
	
	@staticmethod
	@abstractmethod
	def can_hold(stationary_type: type[StationaryItem]) -> bool:
		pass

	@classmethod
	def get_holdable_items(cls) -> list[type[StationaryItem]]:
		return [stationary_type for stationary_type in stationary_types if cls.can_hold(stationary_type)]
	
	@staticmethod
	def get_in_room_predicate(room_param: str, item_param: str) -> str:
		return f"{Room.IN_ROOM_RELATION} {room_param} {item_param}"
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		holdable_types = [item_type.get_type_name() for item_type in stationary_types]
		return [Predicate(Room.IN_ROOM_RELATION, [Room.ROOM_PARAM + " - " + Room.TYPE_NAME, "{} - (either {})".format(Room.ITEM_PARAM, " ".join(holdable_types))])]
	
	def get_init_conditions(self) -> list[str]:
		init_conditions: list[str] = []
		for item in self.items:
			init_conditions += item.get_init_conditions()
		return init_conditions
	
	def get_pddl_objects(self) -> list[str]:
		objects: list[str] = [self.token_name + " - " + Room.TYPE_NAME]
		for item in self.items:
			objects += item.get_pddl_objects()
		return objects
	
	def get_knowledge_yaml(self, indent: int) -> str:
		yaml = self.yaml_instance.to_yaml(indent)
		for item in self.items:
			yaml += item.get_yaml_instance().to_yaml(indent)
		return yaml

class Kitchen(Room):
	generated = False
	@staticmethod
	def generate_empty() -> Kitchen | None:
		if Kitchen.generated:
			return None
		Kitchen.generated = True
		return Kitchen("the kitchen", "the_kitchen")
	
	@staticmethod
	def can_hold(stationary_type: type[StationaryItem]) -> bool:
		return stationary_type in [Fridge, KitchenSink, Light, Pantry]

	def generate_goal(self, people: list[Person], all_items: list[MovableItem], agent: Agent) -> Goal | None:
		if random.choice([True, False]):
			goal = super().generate_goal(people, all_items, agent)
			if goal is not None:
				return goal
		pantry = next(item for item in self.items if isinstance(item, Pantry))
		fridge = next(item for item in self.items if isinstance(item, Fridge))
		fridge_goal = fridge.generate_special_goal(agent, combined=True)
		pantry_goal = pantry.generate_special_goal(agent, combined=True)
		return Goal(
			"I want to organize the kitchen. " + fridge_goal.description + " " + pantry_goal.description,
			fridge_goal.predicate_list + pantry_goal.predicate_list
		)

class LivingRoom(Room):
	generated = False
	@staticmethod
	def generate_empty() -> LivingRoom | None:
		if LivingRoom.generated:
			return None
		LivingRoom.generated = True
		return LivingRoom("the living room", "the_living_room")
	
	@staticmethod
	def can_hold(stationary_type: type[StationaryItem]) -> bool:
		return stationary_type in [Window, Table, TV, Shelf, Light]

class Bedroom(Room):
	with open(os.path.join(DIR, "items/names.txt")) as f:	
		available_names = f.read().splitlines()
	
	@staticmethod
	def generate_empty() -> Bedroom | None:
		if len(Bedroom.available_names) == 0:
			return None
		name = Bedroom.available_names.pop(random.randrange(len(Bedroom.available_names)))
		return Bedroom(f"{name}'s bedroom", f"{name.lower()}_bedroom")
	
	@staticmethod
	def can_hold(stationary_type: type[StationaryItem]) -> bool:
		return stationary_type in [Window, Table, TV, Shelf, Light]

class LaundryRoom(Room):
	generated = False

	@staticmethod
	def generate_empty() -> LaundryRoom | None:
		if LaundryRoom.generated:
			return None
		LaundryRoom.generated = True
		return LaundryRoom("the laundry room", "the_laundry_room")
	
	@staticmethod
	def can_hold(stationary_type: type[StationaryItem]) -> bool:
		return stationary_type in [Light, Washer, Dryer, LaundryBasket]

class Bathroom(Room):
	generated = False

	@staticmethod
	def generate_empty() -> Bathroom | None:
		if Bathroom.generated:
			return None
		Bathroom.generated = True
		return Bathroom("the bathroom", "the_bathroom")
	
	@staticmethod
	def can_hold(stationary_type: type[StationaryItem]) -> bool:
		return stationary_type in [Light, Sink, Toilet]

room_types: list[type[Room]]

class Agent:
	TYPE_NAME = "agent"
	IN_HAND_RELATION = "in_agent_hand"
	IN_ROOM_RELATION = "agent_in_room"

	def __init__(self, parent: Room) -> None:
		self.token_name = "the_agent"
		self.entity_id = EntityID(self.token_name, Agent.TYPE_NAME)
		self.parent = parent
	
	@staticmethod
	def generate_agent(rooms: list[Room]) -> Agent:
		return Agent(random.choice(rooms))

	@staticmethod
	def get_in_hand_predicate(item_param: str, agent_param: str) -> str:
		return f"{Agent.IN_HAND_RELATION} {item_param} {agent_param}"
	
	@staticmethod
	def get_in_room_predicate(agent_param: str, room_param: str) -> str:
		return f"{Agent.IN_ROOM_RELATION} {agent_param} {room_param}"
	
	@staticmethod
	def get_pddl_domain_predicates() -> list[Predicate]:
		type_list = "(either {})".format(" ".join([t.get_type_name() for t in movable_types]))
		return [
			Predicate(Agent.IN_HAND_RELATION, [f"?a - {type_list}", f"?b - {Agent.TYPE_NAME}"]),
			Predicate(Agent.IN_ROOM_RELATION, [f"?a - {Agent.TYPE_NAME}", f"?b - {Room.TYPE_NAME}"]),
		]
	
	def get_pddl_objects(self) -> list[str]:
		return [self.token_name + " - " + Agent.TYPE_NAME]
	
	def get_init_conditions(self) -> list[str]:
		return [self.get_in_room_predicate(self.token_name, self.parent.token_name)]
	
	def get_yaml_instance(self) -> Instance:
		return Instance(self.entity_id, [Attribute(Agent.IN_ROOM_RELATION, self.parent.entity_id)])
	
	@staticmethod
	def get_pddl_domain_actions() -> list[Action]:
		type_list = "(either {})".format(" ".join([t.get_type_name() for t in movable_types]))
		param_list = [f"?a - {type_list}", f"?b - {Agent.TYPE_NAME}", f"?c - {Person.TYPE_NAME}", f"?d - {Room.TYPE_NAME}"]
		person_in_room_predicate = Person.get_in_room_predicate("?c", "?d")
		agent_in_room_predicate = Agent.get_in_room_predicate("?b", "?d")
		in_person_hand_predicate = Person.get_in_hand_predicate("?a", "?c")
		in_agent_hand_predicate = Agent.get_in_hand_predicate("?a", "?b")

		return [
			Action(
				"hand_to_person",
				param_list,
				[in_agent_hand_predicate, person_in_room_predicate, agent_in_room_predicate],
				[f"not ({in_agent_hand_predicate})", in_person_hand_predicate]
			),
			Action(
				"take_from_person",
				param_list,
				[in_person_hand_predicate, person_in_room_predicate, agent_in_room_predicate],
				[f"not ({in_person_hand_predicate})", in_agent_hand_predicate]
			),
			Action(
				"move_to_room",
				[f"?a - {Agent.TYPE_NAME}", f"?b - {Room.TYPE_NAME}", f"?c - {Room.TYPE_NAME}"],
				[Agent.get_in_room_predicate("?a", "?b")],
				[f"not ({Agent.get_in_room_predicate('?a', '?b')})", Agent.get_in_room_predicate("?a", "?c")]
			)
		]
	
	def get_description(self) -> str:
		return f"The agent is in {self.parent.name}. "

class SimulationGenerator:
	MAX_ROOMS = random.randint(30, 40)
	MAX_ITEMS = random.randint(140, 160)
	MAX_PEOPLE = random.randint(20, 30)

	def __init__(self, parent_dir: str, num_state_changes: int = 100, state_changes_per_query: int = 10, state_changes_per_goal: int = 20) -> None:
		self.num_state_changes = num_state_changes
		self.state_changes_per_query = state_changes_per_query
		self.state_changes_per_goal = state_changes_per_goal
		self.parent_dir = parent_dir
		self.rooms: list[Room] = []
		self.people: list[Person] = []
		self.description = ""
		
		self.item_type_goal_freq: dict[type[MovableItem], int] = {}
		self.item_goal_freq: dict[MovableItem, int] = {}
		self.item_type_update_freq: dict[type[MovableItem], int] = {}
		self.item_update_freq: dict[MovableItem, int] = {}

		self.room_type_goal_freq: dict[type[Room], int] = {}
		self.room_goal_freq: dict[Room, int] = {}
		self.room_type_update_freq: dict[type[Room], int] = {}
		self.room_update_freq: dict[Room, int] = {}

		self.collective_goal_type_freq: dict[type[CollectiveGoal], int] = {}

		self.movable_items: list[MovableItem] = []
		for movable_type in creatable_movable_types:
			count = 0
			while count < SimulationGenerator.MAX_ITEMS / len(creatable_movable_types):
				item = movable_type.generate_instance()
				if item is None:
					break
				self.movable_items.append(item)
				count += 1
		random.shuffle(self.movable_items)
		
		self.stationary_items: list[StationaryItem] = []
		for room_type in room_types:
			count = 0
			while count < SimulationGenerator.MAX_ROOMS / len(room_types):
				pair = room_type.generate_outline()
				if pair is None:
					break
				room, additional = pair
				count += 1
				self.rooms.append(room)
				self.movable_items += additional
				self.stationary_items += room.items
		
		while len(self.people) < SimulationGenerator.MAX_PEOPLE:
			person = Person.generate_person(self.rooms)
			if person is None:
				break
			self.people.append(person)
		self.agent = Agent.generate_agent(self.rooms)
		
		remaining_movables = self.movable_items.copy()
		while remaining_movables:
			random.shuffle(self.rooms)
			for room in self.rooms:
				room.populate(remaining_movables)
		for room in self.rooms:
			self.description += room.get_description() + "\n\n"
		for person in self.people:
			self.description += person.get_description()
		self.description += self.agent.get_description()

	FRAC_ROOM_UPDATES = 0.5
	FRAC_MOVABLE_UPDATES = 0.3
	FRAC_PERSON_UPDATES = 0.2

	def generate_state_change(self) -> str:
		all_items = self.movable_items.copy()
		usable_rooms, room_probabilities = get_items_and_probabilities(self.room_update_freq, self.room_type_update_freq, self.rooms)
		usable_movables, movable_probabilities = get_items_and_probabilities(self.item_update_freq, self.item_type_update_freq, self.movable_items)
		usable_people = self.people.copy()
		for _ in range(MAX_ITER):
			assert len(usable_rooms) > 0 or len(usable_movables) > 0 or len(usable_people) > 0
			choice = np.random.choice(np.arange(3), p=[SimulationGenerator.FRAC_ROOM_UPDATES, SimulationGenerator.FRAC_MOVABLE_UPDATES, SimulationGenerator.FRAC_PERSON_UPDATES])
			if choice == 0:
				if len(usable_rooms) == 0:
					continue
				room_probabilities = normalize_probabilities(room_probabilities)
				idx = np.random.choice(np.arange(len(usable_rooms)), p=room_probabilities)
				room_probabilities.pop(idx)
				room = usable_rooms.pop(idx)
				action = room.perform_action(self.people)
				if action is not None:
					self.room_update_freq[room] = self.room_update_freq.get(room, 0) + 1
					self.room_type_update_freq[type(room)] = self.room_type_update_freq.get(type(room), 0) + 1
					return action
			elif choice == 1:
				if len(usable_movables) == 0:
					continue
				movable_probabilities = normalize_probabilities(movable_probabilities)
				idx = np.random.choice(np.arange(len(usable_movables)), p=movable_probabilities)
				movable_probabilities.pop(idx)
				item = usable_movables.pop(idx)
				action = item.perform_action(self.people)
				if action is not None:
					self.item_update_freq[item] = self.item_update_freq.get(item, 0) + 1
					self.item_type_update_freq[type(item)] = self.item_type_update_freq.get(type(item), 0) + 1
					return action
			elif choice == 2:
				if len(usable_people) == 0:
					continue
				action = usable_people.pop(random.randrange(len(usable_people))).perform_action(all_items)
				if action is not None:
					return action
		raise Exception("Unable to generate state change")
	
	FRAC_ROOM_GOALS = 0.35
	FRAC_COLLECTIVE_GOALS = 0.1
	FRAC_MOVABLE_GOALS = 0.3
	FRAC_PERSON_GOALS = 0.25

	def generate_goal(self) -> Goal:
		all_items = self.movable_items.copy()
		all_stationary = self.stationary_items.copy()
		usable_rooms, room_probabilities = get_items_and_probabilities(self.room_goal_freq, self.room_type_goal_freq, self.rooms)
		usable_movables, movable_probabilities = get_items_and_probabilities(self.item_goal_freq, self.item_type_goal_freq, self.movable_items)
		usable_collectives, collective_probabilities = get_items_and_probabilities(self.collective_goal_type_freq, {}, collective_goal_types)
		usable_people = self.people.copy()
		for _ in range(MAX_ITER):
			assert len(usable_rooms) > 0 or len(usable_movables) > 0 or len(usable_people) > 0
			choice = np.random.choice(np.arange(4), p=[SimulationGenerator.FRAC_ROOM_GOALS, SimulationGenerator.FRAC_COLLECTIVE_GOALS, SimulationGenerator.FRAC_MOVABLE_GOALS, SimulationGenerator.FRAC_PERSON_GOALS])
			if choice == 0:
				if len(usable_rooms) == 0:
					continue
				room_probabilities = normalize_probabilities(room_probabilities)
				idx = np.random.choice(np.arange(len(usable_rooms)), p=room_probabilities)
				room_probabilities.pop(idx)
				room = usable_rooms.pop(idx)
				goal = room.generate_goal(self.people, all_items, self.agent)
				if goal is not None:
					self.room_goal_freq[room] = self.room_goal_freq.get(room, 0) + 1
					self.room_type_goal_freq[type(room)] = self.room_type_goal_freq.get(type(room), 0) + 1
					return goal
			elif choice == 1:
				if len(usable_collectives) == 0:
					continue
				collective_probabilities = normalize_probabilities(collective_probabilities)
				idx = np.random.choice(np.arange(len(usable_collectives)), p=collective_probabilities)
				collective_probabilities.pop(idx)
				goal_type = usable_collectives.pop(idx)
				goal = goal_type.generate_goal(all_items, all_stationary, self.agent)
				if goal is not None:
					self.collective_goal_type_freq[goal_type] = self.collective_goal_type_freq.get(goal_type, 0) + 1
					return goal
			elif choice == 2:
				if len(usable_movables) == 0:
					continue
				movable_probabilities = normalize_probabilities(movable_probabilities)
				idx = np.random.choice(np.arange(len(usable_movables)), p=movable_probabilities)
				movable_probabilities.pop(idx)
				item = usable_movables.pop(idx)
				goal = item.generate_goal(self.people, all_items, self.agent)
				if goal is not None:
					self.item_goal_freq[item] = self.item_goal_freq.get(item, 0) + 1
					self.item_type_goal_freq[type(item)] = self.item_type_goal_freq.get(type(item), 0) + 1
					return goal
			elif choice == 3:
				if len(usable_people) == 0:
					continue
				goal = usable_people.pop(random.randrange(len(usable_people))).generate_goal(all_items, self.agent)
				if goal is not None:
					return goal
		raise Exception("Unable to generate goal")
	
	def generate_query_answer(self) -> tuple[str, str]:
		if random.choice([True, False]):
			return random.choice(self.movable_items).generate_query_answer()
		return random.choice(self.rooms).generate_query_answer()
	
	def run(self) -> None:
		os.makedirs(self.parent_dir, exist_ok=True)
		with open(os.path.join(self.parent_dir, "initial_state.txt"), "w") as f:
			f.write(self.description)
		predicate_names, domain_pddl = self.generate_domain_pddl()
		with open(os.path.join(self.parent_dir, "predicate_names.txt"), "w") as f:
			f.write("\n".join(predicate_names))
		with open(os.path.join(self.parent_dir, "domain.pddl"), "w") as f:
			f.write(domain_pddl)
		with open(os.path.join(self.parent_dir, "problem.pddl"), "w") as f:
			f.write(self.generate_problem_pddl())
		with open(os.path.join(self.parent_dir, "knowledge.yaml"), "w") as f:
			f.write(self.generate_knowledge_yaml())
		
		time_step = 0
		for i in range(self.num_state_changes):
			curr_dir = os.path.join(self.parent_dir, f"time_{time_step:04d}_state_change")
			os.makedirs(curr_dir, exist_ok=True)
			with open(os.path.join(curr_dir, "state_change.txt"), "w") as f:
				f.write(self.generate_state_change())
			with open(os.path.join(curr_dir, "problem.pddl"), "w") as f:
				f.write(self.generate_problem_pddl())
			with open(os.path.join(curr_dir, "knowledge.yaml"), "w") as f:
				f.write(self.generate_knowledge_yaml())
			time_step += 1
			if (i + 1) % self.state_changes_per_query == 0:
				curr_dir = os.path.join(self.parent_dir, f"time_{time_step:04d}_query")
				os.makedirs(curr_dir, exist_ok=True)
				query, true_answer = self.generate_query_answer()
				with open(os.path.join(curr_dir, "query.txt"), "w") as f:
					f.write(query)
				with open(os.path.join(curr_dir, "answer.txt"), "w") as f:
					f.write(true_answer)
				time_step += 1
			if (i + 1) % self.state_changes_per_goal == 0:
				curr_dir = os.path.join(self.parent_dir, f"time_{time_step:04d}_goal")
				os.makedirs(curr_dir, exist_ok=True)
				problem_pddl = self.generate_problem_pddl(with_goal=True)
				goal = self.generate_goal()
				with open(os.path.join(curr_dir, "goal.txt"), "w") as f:
					f.write(goal.description)
				with open(os.path.join(curr_dir, "problem.pddl"), "w") as f:
					f.write(problem_pddl.format(str(goal)))
				with open(os.path.join(curr_dir, "knowledge.yaml"), "w") as f:
					f.write(self.generate_knowledge_yaml())
				time_step += 1
	
	@staticmethod
	def generate_domain_pddl() -> tuple[list[str], str]:
		predicates: list[Predicate] = Person.get_pddl_domain_predicates() + Room.get_pddl_domain_predicates()
		actions: list[Action] = []
		required_types: list[str] = [Person.TYPE_NAME, Room.TYPE_NAME, Agent.TYPE_NAME]
		for item_type in item_types:
			predicates += item_type.get_pddl_domain_predicates()
			actions += item_type.get_pddl_domain_actions()
			required_types += item_type.get_required_types()
		
		required_types = sorted(required_types, key=len)

		predicates += Agent.get_pddl_domain_predicates()
		actions += Agent.get_pddl_domain_actions()

		predicate_names = [predicate.name for predicate in predicates]

		formatted_predicates = [str(predicate) for predicate in predicates]
		formatted_actions = [str(action) for action in actions]

		return predicate_names, \
				"(define (domain simulation)\n" \
					+ "\t(:requirements :typing :negative-preconditions)\n" \
					+ "\t(:types\n" \
						+ "\t\t{}\n".format("\n\t\t".join(required_types)) \
					+ "\t)\n" \
					+ "\t(:predicates\n" \
						+ "\t\t{}\n".format("\n\t\t".join(formatted_predicates)) \
					+ "\t)\n\n" \
					+ "{}".format("\n".join(formatted_actions)) \
				+ ")\n"
	
	def generate_problem_pddl(self, with_goal: bool = False) -> str:
		objects: list[str] = []
		init_conditions: list[str] = []

		objects += self.agent.get_pddl_objects()
		init_conditions += self.agent.get_init_conditions()

		for person in self.people:
			objects += person.get_pddl_objects()
			init_conditions += person.get_init_conditions()

		for room in self.rooms:
			objects += room.get_pddl_objects()
			init_conditions += room.get_init_conditions()
		
		for item in self.movable_items:
			objects += item.get_pddl_objects()
			init_conditions += item.get_init_conditions()
		
		for entity in static_entities:
			objects.append(f"{entity.entity_id.name} - {entity.entity_id.concept}")
		
		return "(define (problem simulation-a)\n" \
					+ "\t(:domain simulation)\n" \
					+ "\t(:objects\n" \
						+ "\t\t{}\n".format("\n\t\t".join(objects)) \
					+ "\t)\n" \
					+ "\t(:init\n" \
						+ "\t\t({})\n".format(")\n\t\t(".join(init_conditions)) \
					+ "\t)\n" \
					+ ("{}" if with_goal else "") \
				+ ")\n"
	
	def generate_knowledge_yaml(self) -> str:
		yaml = "version: 1\nentities:\n"
		for room in self.rooms:
			yaml += room.get_knowledge_yaml(1)
		for item in self.movable_items:
			yaml += item.get_yaml_instance().to_yaml(1)
		for item in static_entities:
			yaml += item.to_yaml(1)
		for person in self.people:
			yaml += person.get_yaml_instance().to_yaml(1)
		yaml += self.agent.get_yaml_instance().to_yaml(1)
		return yaml


class Simulation:
	def __init__(self, parent_dir: str) -> None:
		self.domain_path = os.path.join(parent_dir, "domain.pddl")
		self.initial_knowledge_path = os.path.join(parent_dir, "knowledge.yaml")
		with open(os.path.join(parent_dir, "initial_state.txt")) as f:
			self.initial_state = f.read()
		with open(os.path.join(parent_dir, "predicate_names.txt")) as f:
			self.predicate_names = f.read().splitlines()
		with open(self.domain_path) as f:
			self.domain_pddl = f.read()
		with open(os.path.join(parent_dir, "problem.pddl")) as f:
			self.initial_problem_pddl = f.read()
		with open(self.initial_knowledge_path) as f:
			self.initial_knowledge_yaml = f.read()
		
		time_steps = os.listdir(parent_dir)
		time_steps.remove("initial_state.txt")
		time_steps.remove("predicate_names.txt")
		time_steps.remove("domain.pddl")
		time_steps.remove("problem.pddl")
		time_steps.remove("knowledge.yaml")
		time_steps.sort()

		self.num_time_steps = len(time_steps)
		self.time_steps: list[dict[str, Any]] = []

		for i, time_step in enumerate(time_steps):
			curr_dir = os.path.join(parent_dir, time_step)
			curr_data: dict[str, Any] = {"time" : i}
			if time_step.endswith("query"):
				curr_data["type"] = "query"
				with open(os.path.join(curr_dir, "query.txt")) as f:
					curr_data["query"] = f.read()
				with open(os.path.join(curr_dir, "answer.txt")) as f:
					curr_data["answer"] = f.read()
			elif time_step.endswith("state_change"):
				curr_data["type"] = "state_change"
				with open(os.path.join(curr_dir, "state_change.txt")) as f:
					curr_data["state_change"] = f.read()
				curr_data["problem_path"] = os.path.join(curr_dir, "problem.pddl")
				with open(curr_data["problem_path"]) as f:
					curr_data["problem_pddl"] = f.read()
				curr_data["knowledge_path"] = os.path.join(curr_dir, "knowledge.yaml")
				with open(curr_data["knowledge_path"]) as f:
					curr_data["knowledge_yaml"] = f.read()
			elif time_step.endswith("goal"):
				curr_data["type"] = "goal"
				with open(os.path.join(curr_dir, "goal.txt")) as f:
					curr_data["goal"] = f.read()
				curr_data["problem_path"] = os.path.join(curr_dir, "problem.pddl")
				with open(curr_data["problem_path"]) as f:
					curr_data["problem_pddl"] = f.read()
				curr_data["knowledge_path"] = os.path.join(curr_dir, "knowledge.yaml")
				with open(curr_data["knowledge_path"]) as f:
					curr_data["knowledge_yaml"] = f.read()
				try:
					true_plan_path = os.path.join(curr_dir, "true_plan.pddl")
					with open(true_plan_path) as f:
						curr_data["true_plan_path"] = true_plan_path
						curr_data["true_plan_pddl"] = f.read()
					with open(os.path.join(curr_dir, "true_plan.time")) as f:
						curr_data["true_plan_time"] = float(f.read())
				except:
					pass
			else:
				raise Exception("Invalid directory:", time_step)
			self.time_steps.append(curr_data)
		
		self.curr_time_step = -1
	
	def __iter__(self):
		return self
	
	def __next__(self):
		self.curr_time_step += 1
		if self.curr_time_step >= self.num_time_steps:
			raise StopIteration
		return self.time_steps[self.curr_time_step]

T = TypeVar('T')
def get_concrete_subtypes(initial_type: type[T]) -> list[type[T]]:
	found_types: list[type] = [initial_type]
	concrete_subtypes: set[type] = set()
	while len(found_types) > 0:
		curr_type = found_types.pop()
		if not isabstract(curr_type):
			concrete_subtypes.add(curr_type)
		found_types.extend(curr_type.__subclasses__())
	return list(concrete_subtypes)

item_types = get_concrete_subtypes(RoomItem)
movable_types = get_concrete_subtypes(MovableItem)
creatable_movable_types = [movable_type for movable_type in movable_types if not issubclass(movable_type, AccompanyingItem)]
stationary_types = get_concrete_subtypes(StationaryItem)
room_types = get_concrete_subtypes(Room)
collective_goal_types = get_concrete_subtypes(CollectiveGoal)

static_entities: list[Instance] = []
for item_type in item_types:
	static_entities += item_type.get_static_entities()

if __name__ == "__main__":
	generator = SimulationGenerator("experiment/domain", num_state_changes=100, state_changes_per_query=300, state_changes_per_goal=5)
	generator.run()